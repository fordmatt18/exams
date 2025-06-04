"""
Core logic for the Benders decomposition algorithm, including:
- Task function for solving individual student pattern subproblems (for parallel execution).
- Function to calculate the expected cost of a single student pattern given a master schedule.
- Function to generate an initial master schedule.
"""
import os
import random
from pyomo.opt import SolverStatus, TerminationCondition # For enums used in task return

# Assuming model_builder and solver_utils are in the same directory or PYTHONPATH
from model_builder import build_student_pattern_subproblem_model
from solver_utils import solve_subproblem_and_get_duals

def solve_student_pattern_subproblem_task(args_tuple):
    """
    Worker function to solve a single student pattern subproblem.
    Designed to be used with concurrent.futures.ProcessPoolExecutor.map().

    Args:
        args_tuple (tuple): A tuple containing all necessary arguments:
            - student_pattern_tuple (tuple): The specific student pattern (tuple of exam IDs).
            - all_time_slots_set (list/set): Global set of time slots.
            - master_schedule_solution_dict (dict): Current x_et^(k) solution from MP.
            - subproblem_solver_name (str): Name of the solver for the SP.
            - print_subproblem_details_flag (bool): If True, print details during SP solve.
            - subproblem_objective_weight (float): Weight for this SP's objective.
            - num_exams_in_pattern (int): The number of exams in student_pattern_tuple (k).
            - precomputed_canonical_schedule_costs_for_k (dict):
                Maps canonical schedule assignments (for k placeholder exams) to their costs (U_pi).
                Example key: (('PEX0', slotA), ('PEX1', slotB)), Value: cost.

    Returns:
        dict: A dictionary containing results for this subproblem:
            - 'student_pattern_tuple': The input student_pattern_tuple.
            - 'cost': The optimal objective value (f_s^(k)). float('inf') on failure.
            - 'duals': Dictionary of linking constraint duals (lambda_s,e,t^(k)). Empty on failure.
            - 'status': Solver status enum.
            - 'term_cond': Termination condition enum.
            - 'error_message': String error message if any, else None.
            - 'subproblem_failed_flag': Boolean, True if SP could not be solved to optimality.
    """
    (student_pattern_tuple, all_time_slots_set, master_schedule_solution_dict,
     subproblem_solver_name, print_subproblem_details_flag, subproblem_objective_weight,
     num_exams_in_pattern, precomputed_canonical_schedule_costs_for_k) = args_tuple

    exams_in_student_pattern = list(student_pattern_tuple)
    student_pattern_id_str = "_".join(exams_in_student_pattern) if exams_in_student_pattern else "empty_pattern"

    # Base case: if a student pattern has no exams, its cost is 0.
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
        # Construct the specific U_pi mapping for *this student pattern's actual exams*
        # from the precomputed_canonical_schedule_costs_for_k (which use placeholder exams).
        personal_schedule_to_cost_map_for_this_pattern = {}
        for canonical_schedule_assignment_tuple, cost_value in precomputed_canonical_schedule_costs_for_k.items():
            # canonical_schedule_assignment_tuple is like (('PEX0', slotA), ('PEX1', slotB), ...)
            if len(canonical_schedule_assignment_tuple) != num_exams_in_pattern:
                # Should not happen if precomputed_canonical_schedule_costs_for_k is correctly filtered for k
                continue

            # Map placeholder exams in canonical_schedule to actual exams in student_pattern_tuple
            current_personal_schedule_for_student_pattern = []
            valid_mapping = True
            for i in range(num_exams_in_pattern):
                # placeholder_exam_in_canonical = canonical_schedule_assignment_tuple[i][0] # e.g., 'PEX0'
                # actual_exam_for_this_pattern = exams_in_student_pattern[i] # e.g., 'MATH101'
                # slot_for_this_exam = canonical_schedule_assignment_tuple[i][1] # e.g., slotA
                # current_personal_schedule_for_student_pattern.append(
                #     (actual_exam_for_this_pattern, slot_for_this_exam)
                # )
                # The above assumes canonical_placeholder_exams_list was sorted like exams_in_student_pattern.
                # A safer way if canonical_placeholder_exams_list was ['PEX0', 'PEX1', ...]
                # and exams_in_student_pattern is sorted ['ActualE1', 'ActualE2', ...]
                # then PEX0 maps to ActualE1, PEX1 to ActualE2.
                # The canonical_schedule_assignment_tuple is already ordered by placeholder exam index.
                actual_exam = exams_in_student_pattern[i] # exams_in_student_pattern is sorted
                slot = canonical_schedule_assignment_tuple[i][1] # Slot for the i-th placeholder exam
                current_personal_schedule_for_student_pattern.append((actual_exam, slot))


            personal_schedule_to_cost_map_for_this_pattern[
                tuple(current_personal_schedule_for_student_pattern)
            ] = cost_value

        if not personal_schedule_to_cost_map_for_this_pattern and num_exams_in_pattern > 0:
            # This can happen if k > num_slots and allow_slot_reuse is False during canonical generation.
            error_msg = (f"No personal schedules constructed from canonical for pattern {student_pattern_id_str} "
                         f"(k={num_exams_in_pattern}). Canonical dict size: {len(precomputed_canonical_schedule_costs_for_k)}.")
            # This is not necessarily an error if it's a valid scenario (e.g., impossible to schedule k exams in T slots without reuse)
            # Treat as infeasible for this SP.
            return {
                'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
                'status': SolverStatus.warning, 'term_cond': TerminationCondition.infeasible,
                'error_message': error_msg, 'subproblem_failed_flag': True
            }

        # Build the Pyomo subproblem model
        student_subproblem_model = build_student_pattern_subproblem_model(
            exams_in_student_pattern, all_time_slots_set, master_schedule_solution_dict,
            personal_schedule_to_cost_map_for_this_pattern,
            student_pattern_id_str=student_pattern_id_str,
            subproblem_objective_weight=subproblem_objective_weight
        )

        if student_subproblem_model is None: # Model creation failed (e.g. no assignments)
            error_msg = (f"SP Model creation failed for pattern {student_pattern_id_str} (k={num_exams_in_pattern}), "
                         "likely due to no valid personal schedules.")
            return {
                'student_pattern_tuple': student_pattern_tuple, 'cost': float('inf'), 'duals': {},
                'status': SolverStatus.error, 'term_cond': TerminationCondition.error,
                'error_message': error_msg, 'subproblem_failed_flag': True
            }

        if print_subproblem_details_flag:
            # Es_pattern is now EXAMS_IN_PATTERN, Pi_idx is PERSONAL_SCHEDULE_INDICES
            num_personal_schedules = len(student_subproblem_model.PERSONAL_SCHEDULE_INDICES) if hasattr(student_subproblem_model, 'PERSONAL_SCHEDULE_INDICES') else 0
            print(f"SP_worker {student_pattern_id_str} (PID {os.getpid()}): "
                  f"{len(student_subproblem_model.EXAMS_IN_PATTERN)} exams, "
                  f"{num_personal_schedules} personal schedules (y_pi vars), "
                  f"weight {subproblem_objective_weight:.2f}, using {subproblem_solver_name}")

        # Solve the subproblem
        _sp_results_obj, sp_objective_value, linking_duals, sp_status, sp_term_cond = \
            solve_subproblem_and_get_duals(
                student_subproblem_model, solver_name=subproblem_solver_name, suppress_solver_output=True
            )

        # Determine if the subproblem solution is valid for Benders cut
        sp_failed = (sp_objective_value is None or sp_objective_value == float('inf') or
                     sp_term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible])
        # For Benders, we typically need optimal SPs. Feasible might be okay if objective is finite.
        # If infeasible, objective is inf, duals might not be useful for standard optimality cuts.

        return {
            'student_pattern_tuple': student_pattern_tuple,
            'cost': sp_objective_value if not sp_failed else float('inf'),
            'duals': linking_duals if not sp_failed and linking_duals else {}, # Ensure duals is a dict
            'status': sp_status,
            'term_cond': sp_term_cond,
            'error_message': None, # Python exception handled below
            'subproblem_failed_flag': sp_failed
        }
    except Exception as e:
        # Catch any unexpected Python errors during the task execution
        error_msg = f"Python exception in worker for pattern {student_pattern_id_str}: {type(e).__name__} - {e}"
        # import traceback # Optionally include full traceback in error_msg
        # error_msg += f"\n{traceback.format_exc()}"
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
    This involves setting up and solving one subproblem instance.
    Used for calculating 'before' and 'after' total objective values.

    Args:
        student_pattern_tuple (tuple): The student pattern (tuple of exam IDs).
        all_time_slots_set (list/set): Global set of time slots.
        master_schedule_solution_dict (dict): Master schedule x_et values.
        subproblem_solver_name (str): Solver for the SP.
        num_exams_in_pattern (int): Number of exams in this pattern.
        precomputed_canonical_schedule_costs_for_k (dict): Canonical costs for k exams.
        print_details (bool): If True, prints detailed messages.

    Returns:
        float: The minimum expected cost for this student pattern. float('inf') on error/infeasibility.
    """
    # This function essentially runs one instance of the subproblem_task logic,
    # but directly, not through the parallel executor.
    # We can call solve_student_pattern_subproblem_task internally for consistency.

    args_for_task = (
        student_pattern_tuple, all_time_slots_set, master_schedule_solution_dict,
        subproblem_solver_name, print_details, # print_details acts as print_sp_details_flag here
        1.0, # subproblem_objective_weight is 1.0 for individual cost calculation
        num_exams_in_pattern, precomputed_canonical_schedule_costs_for_k
    )

    result_dict = solve_student_pattern_subproblem_task(args_for_task)

    if print_details and result_dict['error_message']:
        print(f"Evaluation for pattern {student_pattern_tuple}: {result_dict['error_message']}")
    if print_details and result_dict['subproblem_failed_flag']:
        print(f"Evaluation for pattern {student_pattern_tuple} failed. Cost: {result_dict['cost']}, TermCond: {result_dict['term_cond']}")

    return result_dict['cost'] # This will be float('inf') if failed

def generate_initial_master_schedule(all_exams_set, all_time_slots_set, strategy="spread"):
    """
    Generates an initial feasible schedule (x_et values) for the master problem.

    Args:
        all_exams_set (list/set): Set of all exam IDs.
        all_time_slots_set (list/set): Set of all time slot indices.
        strategy (str): Method for generating the initial schedule:
                        "spread": Distributes exams across available slots.
                        "first_slot": Assigns all exams to the first slot.
                        "random": Assigns exams to random slots.

    Returns:
        dict: A dictionary mapping (exam_id, ts_id) to 1.0 or 0.0, representing
              the initial x_et values. Returns empty dict if no time slots.
    """
    initial_master_schedule_dict = {} # x_et values
    # Ensure exams are processed in a consistent order for reproducibility if strategy depends on order
    sorted_exam_list = sorted(list(all_exams_set))
    num_available_slots = len(all_time_slots_set)

    if not all_time_slots_set or num_available_slots == 0:
        print("Warning: No time slots available for generating initial schedule.")
        return initial_master_schedule_dict # No schedule possible

    # Ensure time slots are also in a list for indexing
    time_slots_list = sorted(list(all_time_slots_set))

    for i, exam_id in enumerate(sorted_exam_list):
        if strategy == "first_slot":
            slot_to_assign = time_slots_list[0]
        elif strategy == "spread":
            # Cycle through available slots
            slot_to_assign = time_slots_list[i % num_available_slots]
        elif strategy == "random":
            slot_to_assign = random.choice(time_slots_list)
        else: # Default to spread if strategy is unknown
            print(f"Warning: Unknown initial schedule strategy '{strategy}'. Defaulting to 'spread'.")
            slot_to_assign = time_slots_list[i % num_available_slots]

        # Set x_et = 1 for the assigned (exam, slot), and 0 for others for this exam
        for ts_id in all_time_slots_set:
            initial_master_schedule_dict[(exam_id, ts_id)] = 1.0 if ts_id == slot_to_assign else 0.0

    return initial_master_schedule_dict