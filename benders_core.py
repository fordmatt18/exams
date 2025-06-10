"""
Core logic for the Benders decomposition algorithm, including:
- Task function for solving individual student pattern subproblems (for parallel execution).
- Function to calculate the expected cost of a single student pattern given a master schedule.
- Function to generate an initial master schedule.
"""
import os
import random
from pyomo.opt import SolverStatus, TerminationCondition # For enums used in task return
import logging

# Assuming model_builder and solver_utils are in the same directory or PYTHONPATH
from model_builder import build_student_pattern_subproblem_model
from solver_utils import solve_subproblem_and_get_duals

# --- Stateful Worker Global Variables ---
# These will be populated once per worker process by the initializer.
worker_sp_template_model = None
worker_assigned_patterns = None
worker_assigned_weights = None
worker_placeholder_exams = None
worker_all_time_slots = None
worker_solver_name = None # <-- ADDED: To store the solver name

def init_worker_sp(num_exams, time_slots, canonical_costs, work_queue, solver_name): # <-- MODIFIED: Added solver_name
    """
    An initializer function for each worker in the process pool.
    It runs ONCE when the worker process is created.
    1. Builds the template subproblem model.
    2. Pulls its own unique, fixed chunk of work from the shared queue.
    3. Stores all this information in its own global state.
    """
    global worker_sp_template_model, worker_assigned_patterns, worker_assigned_weights
    global worker_placeholder_exams, worker_all_time_slots, worker_solver_name # <-- MODIFIED

    pid = os.getpid()
    logging.info(f"Initializing worker PID {pid}...")
    
    try:
        # Step 1: Store static info and build the template model
        worker_placeholder_exams = [f"__PEX__{i}" for i in range(num_exams)]
        worker_all_time_slots = time_slots
        worker_solver_name = solver_name # <-- ADDED: Store the solver name
        
        worker_sp_template_model = build_student_pattern_subproblem_model(
            worker_placeholder_exams,
            worker_all_time_slots,
            {},  # Master schedule is empty during initialization
            canonical_costs,
            student_pattern_id_str=f"template_k{num_exams}_pid{pid}"
        )
        
        if worker_sp_template_model is None:
            logging.error(f"Worker PID {pid} FAILED to build template SP model.")
            return

        # Step 2: Get this worker's assigned chunk of work from the queue
        chunk_data = work_queue.get()
        worker_assigned_patterns = chunk_data['patterns']
        worker_assigned_weights = chunk_data['weights']
        
        logging.info(f"Worker PID {pid} successfully initialized with solver '{worker_solver_name}'. Assigned {len(worker_assigned_patterns)} patterns.")

    except Exception as e:
        logging.critical(f"CRITICAL ERROR during worker PID {pid} initialization: {e}", exc_info=True)


def solve_my_assigned_chunk_task(master_schedule_solution_dict):
    """
    Worker function that solves its PRE-ASSIGNED chunk of subproblems.
    The only argument it receives per Benders iteration is the new master schedule.
    """
    global worker_sp_template_model, worker_assigned_patterns, worker_assigned_weights
    global worker_placeholder_exams, worker_all_time_slots, worker_solver_name # <-- MODIFIED

    pid = os.getpid()
    if worker_sp_template_model is None or worker_assigned_patterns is None:
        logging.error(f"Worker PID {pid} was not initialized correctly. Cannot solve.")
        return []

    chunk_results = []
    for student_pattern_tuple in worker_assigned_patterns:
        try:
            # 1. Update the mutable master_schedule_param of the persistent template model.
            for i, placeholder_exam in enumerate(worker_placeholder_exams):
                actual_exam = student_pattern_tuple[i]
                for ts_id in worker_all_time_slots:
                    x_bar_val = master_schedule_solution_dict.get((actual_exam, ts_id), 0.0)
                    worker_sp_template_model.master_schedule_param[placeholder_exam, ts_id] = x_bar_val

            # 2. Solve the updated model, now using the stored solver name
            _sp_results_obj, sp_objective_value, linking_duals, sp_status, sp_term_cond = \
                solve_subproblem_and_get_duals(
                    worker_sp_template_model,
                    solver_name=worker_solver_name, # <-- MODIFIED: Pass the correct solver name
                    suppress_solver_output=True
                )

            sp_failed = (sp_objective_value is None or sp_objective_value == float('inf') or
                         sp_term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible])

            # 3. Map the duals from placeholder exams back to actual exams
            mapped_duals = {}
            if not sp_failed and linking_duals:
                for (placeholder_exam, ts_id), dual_val in linking_duals.items():
                    placeholder_idx = worker_placeholder_exams.index(placeholder_exam)
                    actual_exam = student_pattern_tuple[placeholder_idx]
                    mapped_duals[(actual_exam, ts_id)] = dual_val
            
            # 4. Apply the objective weight
            final_cost = sp_objective_value * worker_assigned_weights.get(student_pattern_tuple, 1.0)

            chunk_results.append({
                'student_pattern_tuple': student_pattern_tuple,
                'cost': final_cost if not sp_failed else float('inf'),
                'duals': mapped_duals,
                'status': sp_status,
                'term_cond': sp_term_cond,
                'error_message': None,
                'subproblem_failed_flag': sp_failed
            })

        except Exception as e:
            error_msg = f"Python exception in worker PID {pid} for pattern {student_pattern_tuple}: {type(e).__name__} - {e}"
            chunk_results.append({
                'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
                'status': SolverStatus.error, 'term_cond': TerminationCondition.error,
                'error_message': error_msg, 'subproblem_failed_flag': True
            })
            
    return chunk_results

# The single-solve task remains unchanged for its use in before/after calculations
def solve_student_pattern_subproblem_task(args_tuple):
    """
    Worker function to solve a single student pattern subproblem.
    NOTE: This is kept for simplicity for the before/after objective calculations.
    """
    (student_pattern_tuple, all_time_slots_set, master_schedule_solution_dict,
     subproblem_solver_name, print_subproblem_details_flag, subproblem_objective_weight,
     num_exams_in_pattern, precomputed_canonical_schedule_costs_for_k) = args_tuple

    exams_in_student_pattern = list(student_pattern_tuple)
    student_pattern_id_str = "_".join(exams_in_student_pattern) if exams_in_student_pattern else "empty_pattern"

    if not exams_in_student_pattern:
        return {
            'student_pattern_tuple': student_pattern_tuple, 'cost': 0.0, 'duals': {},
            'status': SolverStatus.ok, 'term_cond': TerminationCondition.optimal,
            'error_message': None, 'subproblem_failed_flag': False
        }

    if precomputed_canonical_schedule_costs_for_k is None:
        error_msg = f"Canonical schedule costs missing for k={num_exams_in_pattern} in SP task for {student_pattern_id_str}"
        return {
            'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
            'status': SolverStatus.error, 'term_cond': TerminationCondition.error,
            'error_message': error_msg, 'subproblem_failed_flag': True
        }

    try:
        personal_schedule_to_cost_map_for_this_pattern = {}
        for canonical_schedule_assignment_tuple, cost_value in precomputed_canonical_schedule_costs_for_k.items():
            if len(canonical_schedule_assignment_tuple) != num_exams_in_pattern:
                continue
            current_personal_schedule_for_student_pattern = []
            for i in range(num_exams_in_pattern):
                actual_exam = exams_in_student_pattern[i]
                slot = canonical_schedule_assignment_tuple[i][1]
                current_personal_schedule_for_student_pattern.append((actual_exam, slot))
            personal_schedule_to_cost_map_for_this_pattern[
                tuple(current_personal_schedule_for_student_pattern)
            ] = cost_value

        if not personal_schedule_to_cost_map_for_this_pattern and num_exams_in_pattern > 0:
            error_msg = (f"No personal schedules constructed from canonical for pattern {student_pattern_id_str} "
                         f"(k={num_exams_in_pattern}). Canonical dict size: {len(precomputed_canonical_schedule_costs_for_k)}.")
            return {
                'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
                'status': SolverStatus.warning, 'term_cond': TerminationCondition.infeasible,
                'error_message': error_msg, 'subproblem_failed_flag': True
            }

        student_subproblem_model = build_student_pattern_subproblem_model(
            exams_in_student_pattern, all_time_slots_set, master_schedule_solution_dict,
            personal_schedule_to_cost_map_for_this_pattern,
            student_pattern_id_str=student_pattern_id_str,
            subproblem_objective_weight=subproblem_objective_weight
        )

        if student_subproblem_model is None:
            error_msg = (f"SP Model creation failed for pattern {student_pattern_id_str} (k={num_exams_in_pattern}), "
                         "likely due to no valid personal schedules.")
            return {
                'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
                'status': SolverStatus.error, 'term_cond': TerminationCondition.error,
                'error_message': error_msg, 'subproblem_failed_flag': True
            }

        if print_subproblem_details_flag:
            num_personal_schedules = len(student_subproblem_model.PERSONAL_SCHEDULE_INDICES) if hasattr(student_subproblem_model, 'PERSONAL_SCHEDULE_INDICES') else 0
            logging.info(f"SP_worker {student_pattern_id_str} (PID {os.getpid()}): "
                  f"{len(student_subproblem_model.EXAMS_IN_PATTERN)} exams, "
                  f"{num_personal_schedules} personal schedules (y_pi vars), "
                  f"weight {subproblem_objective_weight:.2f}, using {subproblem_solver_name}")

        _sp_results_obj, sp_objective_value, linking_duals, sp_status, sp_term_cond = \
            solve_subproblem_and_get_duals(
                student_subproblem_model, solver_name=subproblem_solver_name, suppress_solver_output=True
            )

        sp_failed = (sp_objective_value is None or sp_objective_value == float('inf') or
                     sp_term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible])

        return {
            'student_pattern_tuple': student_pattern_tuple,
            'cost': sp_objective_value if not sp_failed else float('inf'),
            'duals': linking_duals if not sp_failed and linking_duals else {},
            'status': sp_status,
            'term_cond': sp_term_cond,
            'error_message': None,
            'subproblem_failed_flag': sp_failed
        }
    except Exception as e:
        error_msg = f"Python exception in worker for pattern {student_pattern_id_str}: {type(e).__name__} - {e}"
        return {
            'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
            'status': SolverStatus.error, 'term_cond': TerminationCondition.error,
            'error_message': error_msg,
            'subproblem_failed_flag': True
        }

def compute_expected_cost_for_student_pattern(
    student_pattern_tuple,
    all_time_slots_set,
    master_schedule_solution_dict, # x_et values
    subproblem_solver_name,
    num_exams_in_pattern, # k for this pattern
    precomputed_canonical_schedule_costs_for_k, # U_pi for canonical k-exam patterns
    print_details=False # For verbose output during this specific calculation
):
    """
    Calculates the minimum expected cost for a single student pattern given a master schedule.
    """
    args_for_task = (
        student_pattern_tuple, all_time_slots_set, master_schedule_solution_dict,
        subproblem_solver_name, print_details,
        1.0,
        num_exams_in_pattern, precomputed_canonical_schedule_costs_for_k
    )
    result_dict = solve_student_pattern_subproblem_task(args_for_task)
    if print_details and result_dict['error_message']:
        logging.warning(f"Evaluation for pattern {student_pattern_tuple}: {result_dict['error_message']}")
    if print_details and result_dict['subproblem_failed_flag']:
        logging.warning(f"Evaluation for pattern {student_pattern_tuple} failed. Cost: {result_dict['cost']}, TermCond: {result_dict['term_cond']}")
    return result_dict['cost']

def generate_initial_master_schedule(all_exams_set, all_time_slots_set, strategy="spread"):
    """
    Generates an initial feasible schedule (x_et values) for the master problem.
    """
    initial_master_schedule_dict = {}
    sorted_exam_list = sorted(list(all_exams_set))
    num_available_slots = len(all_time_slots_set)
    if not all_time_slots_set or num_available_slots == 0:
        logging.warning("No time slots available for generating initial schedule.")
        return initial_master_schedule_dict
    time_slots_list = sorted(list(all_time_slots_set))
    for i, exam_id in enumerate(sorted_exam_list):
        if strategy == "first_slot":
            slot_to_assign = time_slots_list[0]
        elif strategy == "spread":
            slot_to_assign = time_slots_list[i % num_available_slots]
        elif strategy == "random":
            slot_to_assign = random.choice(time_slots_list)
        else:
            logging.warning(f"Warning: Unknown initial schedule strategy '{strategy}'. Defaulting to 'spread'.")
            slot_to_assign = time_slots_list[i % num_available_slots]
        for ts_id in all_time_slots_set:
            initial_master_schedule_dict[(exam_id, ts_id)] = 1.0 if ts_id == slot_to_assign else 0.0
    return initial_master_schedule_dict