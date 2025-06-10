import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError
import random
import time
import pandas as pd
import os
import concurrent.futures
import traceback
import logging # Import logging
import math # For ceil
import itertools # For chain
import multiprocessing # For the Manager and Queue

# Import from custom modules
import config
from logger_setup import setup_logging
from data_utils import (
    load_and_structure_student_enrollments,
    generate_simulated_enrollment_data
)
from cost_utils import (
    calculate_canonical_schedule_costs,
    calculate_penalty_for_gap,
    generate_canonical_slot_permutations,
    calculate_cost_for_chronological_schedule
)
from model_builder import (
    build_benders_master_problem,
)
from solver_utils import (
    solve_master_problem_persistent,
    solve_pyomo_model
)
from benders_core import (
    init_worker_sp, # <-- IMPORT THE NEW INITIALIZER
    solve_my_assigned_chunk_task, # <-- IMPORT THE NEW STATEFUL TASK
    compute_expected_cost_for_student_pattern,
    generate_initial_master_schedule
)

if __name__ == "__main__":
    # --- Setup Logging FIRST ---
    # ... (This section is unchanged) ...
    config_to_log = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
    setup_logging(config.LOG_FILE_NAME, config.LOG_LEVEL_CONSOLE, config.LOG_LEVEL_FILE, config_to_log)
    
    # --- Benders Algorithm State Variables ---
    # ... (This section is unchanged) ...
    precomputed_canonical_schedule_costs_by_k_exams = {}
    master_problem_build_times = []
    master_problem_solve_times = []
    subproblem_phase_times = []
    lower_bound = -float('inf')
    upper_bound = float('inf')
    lower_bound_history = []
    upper_bound_history = []
    current_benders_iteration = 0
    best_master_schedule_solution = None
    iteration_of_best_master_schedule = -1
    benders_master_model = None
    script_start_time = time.time()
    persistent_master_solver = None
    student_pattern_cost_tracking_dict = {}

    random.seed(config.RANDOM_SEED)
    logging.info(f"Random seed set to: {config.RANDOM_SEED}")

    try:
        # --- Data Loading and Initial Setup ---
        # ... (This section is unchanged) ...
        logging.info(f">>> === Benders for Exam Patterns (CSV: {config.STUDENT_ENROLLMENT_CSV_PATH}) === <<<")
        logging.info(f"Optimizing for patterns with {config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION} exams.")
        logging.info(f"Student pattern subset fraction for MP: {config.STUDENT_PATTERN_SUBSET_FRACTION*100:.1f}%")
        logging.info(f"Reweight SP Objective by Count: {config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT}")
        logging.info(f"Parallel SP Solving: {config.USE_PARALLEL_SUBPROBLEM_SOLVING} (Workers: {config.NUM_SUBPROBLEM_WORKERS if config.NUM_SUBPROBLEM_WORKERS else 'Default'})")
        logging.info(f"Allow Intra-Pattern Conflicts (penalized): {config.ALLOW_INTRA_PATTERN_CONFLICTS}")
        logging.info(f"Relax Master Problem Variables: {config.RELAX_MASTER_PROBLEM_VARIABLES}")
        logging.info(f"Calculate Before/After Objectives: {config.CALCULATE_BEFORE_AFTER_OBJECTIVES}")
        if not os.path.exists(config.STUDENT_ENROLLMENT_CSV_PATH):
            logging.warning(f"{config.STUDENT_ENROLLMENT_CSV_PATH} not found. Creating dummy data.")
            dummy_exams = [f'EXAM{k}' for k in range(1, 11)]
            num_dummy_students = 20
            dummy_data_list = []
            for i in range(1, num_dummy_students + 1):
                student_id = f's{i}'
                if i <= 2 and config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION > 0 and config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION <= len(dummy_exams):
                    exams_taken = random.sample(dummy_exams, k=config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION)
                else:
                    num_exams_for_student = random.randint(max(1, config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION -1), min(len(dummy_exams), config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION + 1))
                    if num_exams_for_student > 0 :
                         exams_taken = random.sample(dummy_exams, k=num_exams_for_student)
                    else:
                         exams_taken = []
                dummy_data_list.append({'student': student_id, 'exams': str(exams_taken)})
            if config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION == 4:
                 dummy_data_list[0]['exams'] = str(['EXAM1', 'EXAM2', 'EXAM3', 'EXAM4'])
                 if num_dummy_students > 1:
                    dummy_data_list[1]['exams'] = str(['EXAM1', 'EXAM2', 'EXAM3', 'EXAM4'])
                 if num_dummy_students > 2:
                    dummy_data_list[2]['exams'] = str(['EXAM5', 'EXAM6', 'EXAM7', 'EXAM8'])
            pd.DataFrame(dummy_data_list).to_csv(config.STUDENT_ENROLLMENT_CSV_PATH, index=False)
            logging.info(f"Dummy {config.STUDENT_ENROLLMENT_CSV_PATH} created.")
        enrollments_by_pattern_size, all_exam_ids, max_exams_per_student_from_data = \
            load_and_structure_student_enrollments(config.STUDENT_ENROLLMENT_CSV_PATH)
        if not enrollments_by_pattern_size or not all_exam_ids:
            logging.error("Exiting due to data loading issues.")
            exit()
        all_time_slots = list(range(config.NUM_TIME_SLOTS))
        if not all_time_slots:
            logging.error("NUM_TIME_SLOTS is 0, no time slots available. Exiting.")
            exit()
        num_exams_for_optimization = config.NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION
        if num_exams_for_optimization not in enrollments_by_pattern_size:
            logging.error(f"No student enrollment patterns found with exactly {num_exams_for_optimization} exams. Exiting.")
            exit()
        target_student_pattern_counts = enrollments_by_pattern_size[num_exams_for_optimization]
        target_student_pattern_list = list(target_student_pattern_counts.keys())
        if not target_student_pattern_list:
            logging.error(f"No unique patterns found with {num_exams_for_optimization} exams after grouping. Exiting.")
            exit()
        num_patterns_to_sample = int(len(target_student_pattern_list) * config.STUDENT_PATTERN_SUBSET_FRACTION)
        if num_patterns_to_sample == 0 and len(target_student_pattern_list) > 0:
            num_patterns_to_sample = 1
        student_patterns_for_benders_optimization = random.sample(target_student_pattern_list, num_patterns_to_sample)
        logging.info(f"Found {len(target_student_pattern_list)} unique patterns with {num_exams_for_optimization} exams.")
        logging.info(f"Selected {len(student_patterns_for_benders_optimization)} patterns for Benders optimization process.")
        if not student_patterns_for_benders_optimization:
            logging.error("No patterns selected for Benders optimization process. Exiting.")
            exit()
        k_for_canonical_generation = num_exams_for_optimization
        logging.info(f"\nPre-generating CANONICAL schedule costs for patterns with {k_for_canonical_generation} exams...")
        time_pregen_start = time.time()
        generate_canonical_slot_permutations.cache_clear()
        calculate_cost_for_chronological_schedule.cache_clear()
        calculate_penalty_for_gap.cache_clear()
        logging.debug("LRU caches for cost_utils cleared before canonical generation.")
        canonical_placeholder_exams = [f"__PEX__{i}" for i in range(k_for_canonical_generation)]
        canonical_schedule_costs_for_target_k = calculate_canonical_schedule_costs(
            canonical_placeholder_exams,
            all_time_slots,
            config.GAP_PENALTY_DICT,
            allow_slot_reuse_within_canonical_schedule=config.ALLOW_INTRA_PATTERN_CONFLICTS
        )
        time_pregen_end = time.time()
        if canonical_schedule_costs_for_target_k is None:
            logging.error(f"CANONICAL schedule cost generation failed for k={k_for_canonical_generation}. Exiting.")
            exit()
        num_canonical_found = len(canonical_schedule_costs_for_target_k)
        if not canonical_schedule_costs_for_target_k and k_for_canonical_generation > 0:
             logging.warning(f"CANONICAL schedule cost generation for k={k_for_canonical_generation} resulted in ZERO schedules.")
             logging.warning(f"  This might be expected if k ({k_for_canonical_generation}) > num_slots ({len(all_time_slots)}) "
                             f"and intra-pattern conflicts (slot reuse) are disallowed.")
        precomputed_canonical_schedule_costs_by_k_exams[k_for_canonical_generation] = canonical_schedule_costs_for_target_k
        logging.info(f"Canonical schedule cost generation for {k_for_canonical_generation} exams complete. "
                     f"Found {num_canonical_found} schedules. Time: {time_pregen_end - time_pregen_start:.3f}s")
        patterns_for_clash_constraints = target_student_pattern_list
        if config.CALCULATE_BEFORE_AFTER_OBJECTIVES:
            logging.info("\nCalculating 'before' objective for ALL target patterns...")
            initial_master_schedule = generate_initial_master_schedule(all_exam_ids, all_time_slots, strategy="spread")
            total_cost_all_target_patterns_before_optimization = 0.0
            canonical_costs_for_current_k = precomputed_canonical_schedule_costs_by_k_exams.get(k_for_canonical_generation)
            if canonical_costs_for_current_k is None and k_for_canonical_generation > 0:
                logging.error(f"Canonical costs for k={k_for_canonical_generation} not found for 'before' calculation. Exiting.")
                exit()
            for p_tuple in target_student_pattern_list:
                per_instance_cost_initial = compute_expected_cost_for_student_pattern(
                    p_tuple, all_time_slots, initial_master_schedule,
                    config.SUBPROBLEM_SOLVER,
                    k_for_canonical_generation,
                    canonical_costs_for_current_k,
                    print_details=config.PRINT_SUBPROBLEM_DETAILS_PER_ITERATION
                )
                actual_student_count_for_pattern = target_student_pattern_counts[p_tuple]
                total_cost_for_this_pattern_group_initial = per_instance_cost_initial * actual_student_count_for_pattern
                student_pattern_cost_tracking_dict[p_tuple] = {
                    'initial_total_cost': total_cost_for_this_pattern_group_initial,
                    'student_count': actual_student_count_for_pattern,
                    'is_in_optimization_subset': (p_tuple in student_patterns_for_benders_optimization)
                }
                if per_instance_cost_initial != float('inf'):
                    total_cost_all_target_patterns_before_optimization += total_cost_for_this_pattern_group_initial
                else:
                    total_cost_all_target_patterns_before_optimization = float('inf')
                    break
            logging.info(f"Overall 'before' objective for ALL {len(target_student_pattern_list)} target patterns: "
                         f"{total_cost_all_target_patterns_before_optimization:.4f}")
        else:
            logging.info("\nSkipping 'before' objective calculation as per configuration.")
            for p_tuple in target_student_pattern_list:
                 student_pattern_cost_tracking_dict[p_tuple] = {
                    'initial_total_cost': float('nan'),
                    'student_count': target_student_pattern_counts[p_tuple],
                    'is_in_optimization_subset': (p_tuple in student_patterns_for_benders_optimization)
                }
        time_mp_build_start = time.time()
        benders_master_model = build_benders_master_problem(
            all_exam_ids,
            all_time_slots,
            student_patterns_for_benders_optimization,
            target_student_pattern_counts,
            config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT,
            config.RELAX_MASTER_PROBLEM_VARIABLES,
            config.ALLOW_INTRA_PATTERN_CONFLICTS,
            patterns_for_clash_constraints
        )
        master_problem_build_times.append(time.time() - time_mp_build_start)
        logging.info(f"Initial Master Problem build time: {master_problem_build_times[-1]:.3f}s")
        use_persistent_solver_local_flag = False
        if config.USE_PERSISTENT_MASTER_PROBLEM_SOLVER:
            try:
                persistent_master_solver = SolverFactory(config.MASTER_PROBLEM_SOLVER, solver_io='python')
                if not persistent_master_solver.available(exception_flag=False):
                    logging.warning(f"Persistent solver {config.MASTER_PROBLEM_SOLVER} not available. Falling back to standard.")
                    persistent_master_solver = None
                else:
                    persistent_master_solver.set_instance(benders_master_model)
                    logging.info(f"Persistent MP solver ({config.MASTER_PROBLEM_SOLVER}) initialized.")
                    use_persistent_solver_local_flag = True
            except Exception as e_pers:
                logging.error(f"Error initializing persistent solver {config.MASTER_PROBLEM_SOLVER}: {e_pers}. Falling back.")
                persistent_master_solver = None
        
        # --- Benders Decomposition Loop ---
        # MODIFICATION START: The logic is split to handle parallel and sequential cases separately.
        # This allows for proper lifecycle management of multiprocessing resources using 'with' statements,
        # which is the primary fix for the deadlock/hanging issue.

        if config.USE_PARALLEL_SUBPROBLEM_SOLVING:
            # The 'with' statements for the Manager and Executor ensure their processes are
            # properly started and shut down, even if errors occur.
            with multiprocessing.Manager() as manager:
                num_workers = config.NUM_SUBPROBLEM_WORKERS if config.NUM_SUBPROBLEM_WORKERS else (os.cpu_count() or 1)
                work_queue = manager.Queue()

                sp_objective_weights = {
                    p_tuple: (target_student_pattern_counts[p_tuple] if config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT else 1.0)
                    for p_tuple in student_patterns_for_benders_optimization
                }
                num_patterns = len(student_patterns_for_benders_optimization)
                chunk_size = math.ceil(num_patterns / num_workers) if num_workers > 0 else num_patterns
                
                if chunk_size > 0:
                    pattern_chunks = [student_patterns_for_benders_optimization[i:i + chunk_size]
                                      for i in range(0, num_patterns, chunk_size)]
                    for chunk in pattern_chunks:
                        work_queue.put({
                            'patterns': chunk,
                            'weights': {p: sp_objective_weights[p] for p in chunk}
                        })

                init_args = (
                    k_for_canonical_generation,
                    all_time_slots,
                    precomputed_canonical_schedule_costs_by_k_exams[k_for_canonical_generation],
                    work_queue,
                    config.SUBPROBLEM_SOLVER
                )
                
                logging.info(f"Initializing ProcessPoolExecutor with {num_workers} stateful workers.")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=init_worker_sp,
                    initargs=init_args
                ) as process_pool_executor:
                    
                    # This is the Benders loop for the PARALLEL case.
                    while current_benders_iteration < config.MAX_BENDERS_ITERATIONS:
                        current_benders_iteration += 1
                        logging.info(f"\n--- Benders Iteration {current_benders_iteration}/{config.MAX_BENDERS_ITERATIONS} ---")

                        logging.info(f"Solving MP ({'Relaxed' if config.RELAX_MASTER_PROBLEM_VARIABLES else 'Integer'})...")
                        iter_mp_solve_start_time = time.time()
                        if use_persistent_solver_local_flag and persistent_master_solver:
                            mp_results, mp_term_cond = solve_master_problem_persistent(persistent_master_solver, benders_master_model, suppress_solver_output=True, iteration_num=current_benders_iteration, print_debug_flag=config.PRINT_MASTER_PROBLEM_DEBUG_PERSISTENT)
                        else:
                            mp_results, mp_term_cond = solve_pyomo_model(benders_master_model, solver_name=config.MASTER_PROBLEM_SOLVER, suppress_solver_output=True)
                        master_problem_solve_times.append(time.time() - iter_mp_solve_start_time)
                        logging.info(f"MP Solve Time: {master_problem_solve_times[-1]:.3f}s")
                        mp_status = mp_results.solver.status if mp_results and mp_results.solver else SolverStatus.error
                        mp_solved_ok = mp_term_cond in [TerminationCondition.optimal, TerminationCondition.feasible]
                        if not mp_solved_ok:
                            logging.error(f"MP failed (Term: {mp_term_cond}, Status: {mp_status}). Stopping Benders loop.")
                            break
                        mp_obj_val_from_model = pyo.value(benders_master_model.Objective, exception=False)
                        if mp_obj_val_from_model is None: mp_obj_val_from_model = lower_bound 
                        lower_bound = max(lower_bound, mp_obj_val_from_model)
                        lower_bound_history.append(lower_bound)
                        current_master_schedule_solution = {(e, t): (pyo.value(benders_master_model.exam_assignment_vars[e, t], exception=False) if benders_master_model.exam_assignment_vars[e, t].value is not None else 0.0) for e in all_exam_ids for t in all_time_slots}
                        logging.info(f"Iter {current_benders_iteration} LB (for OPTIMIZED patterns subset): {lower_bound:.4f}")
                        if config.PRINT_MASTER_SOLUTION_X_BAR_PER_ITERATION:
                            logging.debug("Current Master Schedule Solution (x_bar - non-zero values):")
                            for (e_sol, t_sol), val_sol in current_master_schedule_solution.items():
                                if abs(val_sol) > 1e-6 : logging.debug(f"  x[{e_sol},{t_sol}] = {val_sol:.4f}")
                        
                        logging.info(f"Solving SPs for {len(student_patterns_for_benders_optimization)} patterns (Parallel)...")
                        iter_sp_total_start_time = time.time()
                        
                        map_args = [current_master_schedule_solution] * num_workers
                        results_from_pool = process_pool_executor.map(solve_my_assigned_chunk_task, map_args)
                        all_sp_results = list(itertools.chain.from_iterable(results_from_pool))

                        # The rest of the loop logic (processing results, checking convergence, adding cuts) is identical for both cases.
                        subproblem_results_this_iteration = {} 
                        current_iteration_upper_bound_contribution = 0.0 
                        any_subproblem_failed_this_iteration = False
                        for result_dict in all_sp_results:
                            p_tuple_res = result_dict['student_pattern_tuple']
                            sp_cost = result_dict['cost'] 
                            sp_duals = result_dict['duals']
                            if result_dict['subproblem_failed_flag'] or result_dict['error_message']:
                                logging.warning(f"SP for pattern {p_tuple_res} (k={k_for_canonical_generation}) failed or error. "
                                                f"TermCond: {result_dict['term_cond']}, Error: {result_dict['error_message']}")
                                any_subproblem_failed_this_iteration = True; sp_cost = float('inf'); sp_duals = {}
                            subproblem_results_this_iteration[p_tuple_res] = {'cost': sp_cost, 'duals': sp_duals}
                            if config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT:
                                current_iteration_upper_bound_contribution += sp_cost 
                            else:
                                student_count_for_ub = target_student_pattern_counts[p_tuple_res]
                                weight = sp_objective_weights.get(p_tuple_res, 1.0)
                                per_instance_cost = sp_cost / weight if weight != 0 else 0
                                current_iteration_upper_bound_contribution += per_instance_cost * student_count_for_ub
                        if any_subproblem_failed_this_iteration: current_iteration_upper_bound_contribution = float('inf')
                        subproblem_phase_times.append(time.time() - iter_sp_total_start_time)
                        logging.info(f"Total SP Time ({len(student_patterns_for_benders_optimization)} patterns): {subproblem_phase_times[-1]:.3f}s")
                        if current_iteration_upper_bound_contribution < upper_bound:
                            upper_bound = current_iteration_upper_bound_contribution
                            best_master_schedule_solution = current_master_schedule_solution.copy()
                            iteration_of_best_master_schedule = current_benders_iteration
                        upper_bound_history.append(upper_bound)
                        logging.info(f"Iter {current_benders_iteration} UB (for OPTIMIZED patterns subset): {upper_bound:.4f}" if upper_bound != float('inf') else f"Iter {current_benders_iteration} UB: inf")
                        if upper_bound == float('inf') and lower_bound == float('inf') and current_benders_iteration > 1:
                            logging.warning("Both LB and UB are infinite. Problem might be infeasible or unbounded.")
                        elif upper_bound == float('inf'): logging.warning("UB is infinite this iteration.")
                        elif lower_bound > -float('inf'):
                            gap = upper_bound - lower_bound
                            logging.info(f"Iter {current_benders_iteration} Gap: {gap:.4f}")
                            relative_gap = gap / (abs(upper_bound) + 1e-9) if abs(upper_bound) > 1e-9 else (gap if abs(lower_bound) < 1e-9 else float('inf')) 
                            logging.info(f"Iter {current_benders_iteration} Relative Gap: {relative_gap:.4%}" if relative_gap != float('inf') else f"Iter {current_benders_iteration} Relative Gap: inf")
                            if gap <= config.BENDERS_ABSOLUTE_TOLERANCE or \
                               (relative_gap != float('inf') and relative_gap <= config.BENDERS_RELATIVE_TOLERANCE):
                                logging.info(f"\nConverged in {current_benders_iteration} iterations.")
                                break
                        else:
                             logging.info("LB is -infinite, UB updated.")
                        if not any_subproblem_failed_this_iteration:
                            logging.info(f"Adding cuts for {len(subproblem_results_this_iteration)} patterns...")
                            cut_count_this_iter = 0
                            for p_tuple_cut, sp_info in subproblem_results_this_iteration.items():
                                subproblem_obj_value = sp_info['cost']
                                subproblem_linking_duals = sp_info['duals']
                                if subproblem_obj_value == float('inf') or not subproblem_linking_duals: 
                                    if config.PRINT_SUBPROBLEM_DUALS_PER_ITERATION: logging.debug(f"  Skipping cut for {p_tuple_cut}, obj={subproblem_obj_value}, no/empty duals.")
                                    continue
                                exams_in_pattern_for_cut = list(p_tuple_cut) 
                                benders_cut_variable_terms = []
                                weight_for_cut = sp_objective_weights.get(p_tuple_cut, 1.0)
                                unweighted_subproblem_obj = subproblem_obj_value / weight_for_cut if weight_for_cut != 0 else 0
                                benders_cut_constant_term = unweighted_subproblem_obj
                                for e_c in exams_in_pattern_for_cut: 
                                    if e_c not in all_exam_ids: continue 
                                    for t_c in all_time_slots:
                                        dual_idx = (e_c, t_c)
                                        dual_val = subproblem_linking_duals.get(dual_idx)
                                        x_bar_val = current_master_schedule_solution.get(dual_idx, 0.0)
                                        if isinstance(dual_val, (int, float)):
                                            benders_cut_variable_terms.append(dual_val * benders_master_model.exam_assignment_vars[e_c, t_c])
                                            benders_cut_constant_term -= dual_val * x_bar_val
                                if benders_cut_variable_terms or abs(benders_cut_constant_term) > 1e-9: 
                                    final_cut_rhs_expression = sum(benders_cut_variable_terms) + benders_cut_constant_term
                                    if p_tuple_cut in benders_master_model.STUDENT_PATTERNS_FOR_OPT: 
                                        try:
                                            benders_master_model.BendersCuts.add(
                                                (current_benders_iteration, p_tuple_cut), 
                                                expr=benders_master_model.student_pattern_expected_cost_vars[p_tuple_cut] >= final_cut_rhs_expression
                                            )
                                            cut_count_this_iter += 1
                                        except Exception as add_cut_err: 
                                            logging.error(f"Error adding cut for pattern {p_tuple_cut}, iter {current_benders_iteration}: {add_cut_err}")
                            if cut_count_this_iter == 0 and current_benders_iteration < config.MAX_BENDERS_ITERATIONS: 
                                logging.warning("Warn: No valid cuts added this iteration.")
                            elif cut_count_this_iter > 0: logging.info(f"Added {cut_count_this_iter} cuts.")
                        elif any_subproblem_failed_this_iteration: 
                            logging.warning("Skipping cut generation due to SP failure(s).")
                        if current_benders_iteration >= config.MAX_BENDERS_ITERATIONS: 
                            logging.info("Max Benders iterations reached.")
        else: # This block handles the SEQUENTIAL case.
            # The Benders loop is duplicated here to keep the logic separate and avoid
            # managing multiprocessing resources when they are not needed.
            while current_benders_iteration < config.MAX_BENDERS_ITERATIONS:
                current_benders_iteration += 1
                logging.info(f"\n--- Benders Iteration {current_benders_iteration}/{config.MAX_BENDERS_ITERATIONS} ---")

                logging.info(f"Solving MP ({'Relaxed' if config.RELAX_MASTER_PROBLEM_VARIABLES else 'Integer'})...")
                iter_mp_solve_start_time = time.time()
                if use_persistent_solver_local_flag and persistent_master_solver:
                    mp_results, mp_term_cond = solve_master_problem_persistent(persistent_master_solver, benders_master_model, suppress_solver_output=True, iteration_num=current_benders_iteration, print_debug_flag=config.PRINT_MASTER_PROBLEM_DEBUG_PERSISTENT)
                else:
                    mp_results, mp_term_cond = solve_pyomo_model(benders_master_model, solver_name=config.MASTER_PROBLEM_SOLVER, suppress_solver_output=True)
                master_problem_solve_times.append(time.time() - iter_mp_solve_start_time)
                logging.info(f"MP Solve Time: {master_problem_solve_times[-1]:.3f}s")
                mp_status = mp_results.solver.status if mp_results and mp_results.solver else SolverStatus.error
                mp_solved_ok = mp_term_cond in [TerminationCondition.optimal, TerminationCondition.feasible]
                if not mp_solved_ok:
                    logging.error(f"MP failed (Term: {mp_term_cond}, Status: {mp_status}). Stopping Benders loop.")
                    break
                mp_obj_val_from_model = pyo.value(benders_master_model.Objective, exception=False)
                if mp_obj_val_from_model is None: mp_obj_val_from_model = lower_bound 
                lower_bound = max(lower_bound, mp_obj_val_from_model)
                lower_bound_history.append(lower_bound)
                current_master_schedule_solution = {(e, t): (pyo.value(benders_master_model.exam_assignment_vars[e, t], exception=False) if benders_master_model.exam_assignment_vars[e, t].value is not None else 0.0) for e in all_exam_ids for t in all_time_slots}
                logging.info(f"Iter {current_benders_iteration} LB (for OPTIMIZED patterns subset): {lower_bound:.4f}")
                if config.PRINT_MASTER_SOLUTION_X_BAR_PER_ITERATION:
                    logging.debug("Current Master Schedule Solution (x_bar - non-zero values):")
                    for (e_sol, t_sol), val_sol in current_master_schedule_solution.items():
                        if abs(val_sol) > 1e-6 : logging.debug(f"  x[{e_sol},{t_sol}] = {val_sol:.4f}")
                
                logging.info(f"Solving SPs for {len(student_patterns_for_benders_optimization)} patterns (Sequential)...")
                iter_sp_total_start_time = time.time()
                
                sp_objective_weights = {p_tuple: (target_student_pattern_counts[p_tuple] if config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT else 1.0) for p_tuple in student_patterns_for_benders_optimization}
                seq_work_queue = multiprocessing.Queue() # A dummy queue for the stateful worker pattern
                seq_work_queue.put({'patterns': student_patterns_for_benders_optimization, 'weights': sp_objective_weights})
                init_worker_sp(k_for_canonical_generation, all_time_slots, precomputed_canonical_schedule_costs_by_k_exams[k_for_canonical_generation], seq_work_queue, config.SUBPROBLEM_SOLVER)
                all_sp_results = solve_my_assigned_chunk_task(current_master_schedule_solution)

                # The rest of the loop logic (processing results, checking convergence, adding cuts) is identical for both cases.
                subproblem_results_this_iteration = {} 
                current_iteration_upper_bound_contribution = 0.0 
                any_subproblem_failed_this_iteration = False
                for result_dict in all_sp_results:
                    p_tuple_res = result_dict['student_pattern_tuple']
                    sp_cost = result_dict['cost'] 
                    sp_duals = result_dict['duals']
                    if result_dict['subproblem_failed_flag'] or result_dict['error_message']:
                        logging.warning(f"SP for pattern {p_tuple_res} (k={k_for_canonical_generation}) failed or error. "
                                        f"TermCond: {result_dict['term_cond']}, Error: {result_dict['error_message']}")
                        any_subproblem_failed_this_iteration = True; sp_cost = float('inf'); sp_duals = {}
                    subproblem_results_this_iteration[p_tuple_res] = {'cost': sp_cost, 'duals': sp_duals}
                    if config.REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT:
                        current_iteration_upper_bound_contribution += sp_cost 
                    else:
                        student_count_for_ub = target_student_pattern_counts[p_tuple_res]
                        weight = sp_objective_weights.get(p_tuple_res, 1.0)
                        per_instance_cost = sp_cost / weight if weight != 0 else 0
                        current_iteration_upper_bound_contribution += per_instance_cost * student_count_for_ub
                if any_subproblem_failed_this_iteration: current_iteration_upper_bound_contribution = float('inf')
                subproblem_phase_times.append(time.time() - iter_sp_total_start_time)
                logging.info(f"Total SP Time ({len(student_patterns_for_benders_optimization)} patterns): {subproblem_phase_times[-1]:.3f}s")
                if current_iteration_upper_bound_contribution < upper_bound:
                    upper_bound = current_iteration_upper_bound_contribution
                    best_master_schedule_solution = current_master_schedule_solution.copy()
                    iteration_of_best_master_schedule = current_benders_iteration
                upper_bound_history.append(upper_bound)
                logging.info(f"Iter {current_benders_iteration} UB (for OPTIMIZED patterns subset): {upper_bound:.4f}" if upper_bound != float('inf') else f"Iter {current_benders_iteration} UB: inf")
                if upper_bound == float('inf') and lower_bound == float('inf') and current_benders_iteration > 1:
                    logging.warning("Both LB and UB are infinite. Problem might be infeasible or unbounded.")
                elif upper_bound == float('inf'): logging.warning("UB is infinite this iteration.")
                elif lower_bound > -float('inf'):
                    gap = upper_bound - lower_bound
                    logging.info(f"Iter {current_benders_iteration} Gap: {gap:.4f}")
                    relative_gap = gap / (abs(upper_bound) + 1e-9) if abs(upper_bound) > 1e-9 else (gap if abs(lower_bound) < 1e-9 else float('inf')) 
                    logging.info(f"Iter {current_benders_iteration} Relative Gap: {relative_gap:.4%}" if relative_gap != float('inf') else f"Iter {current_benders_iteration} Relative Gap: inf")
                    if gap <= config.BENDERS_ABSOLUTE_TOLERANCE or \
                       (relative_gap != float('inf') and relative_gap <= config.BENDERS_RELATIVE_TOLERANCE):
                        logging.info(f"\nConverged in {current_benders_iteration} iterations.")
                        break
                else:
                     logging.info("LB is -infinite, UB updated.")
                if not any_subproblem_failed_this_iteration:
                    logging.info(f"Adding cuts for {len(subproblem_results_this_iteration)} patterns...")
                    cut_count_this_iter = 0
                    for p_tuple_cut, sp_info in subproblem_results_this_iteration.items():
                        subproblem_obj_value = sp_info['cost']
                        subproblem_linking_duals = sp_info['duals']
                        if subproblem_obj_value == float('inf') or not subproblem_linking_duals: 
                            if config.PRINT_SUBPROBLEM_DUALS_PER_ITERATION: logging.debug(f"  Skipping cut for {p_tuple_cut}, obj={subproblem_obj_value}, no/empty duals.")
                            continue
                        exams_in_pattern_for_cut = list(p_tuple_cut) 
                        benders_cut_variable_terms = []
                        weight_for_cut = sp_objective_weights.get(p_tuple_cut, 1.0)
                        unweighted_subproblem_obj = subproblem_obj_value / weight_for_cut if weight_for_cut != 0 else 0
                        benders_cut_constant_term = unweighted_subproblem_obj
                        for e_c in exams_in_pattern_for_cut: 
                            if e_c not in all_exam_ids: continue 
                            for t_c in all_time_slots:
                                dual_idx = (e_c, t_c)
                                dual_val = subproblem_linking_duals.get(dual_idx)
                                x_bar_val = current_master_schedule_solution.get(dual_idx, 0.0)
                                if isinstance(dual_val, (int, float)):
                                    benders_cut_variable_terms.append(dual_val * benders_master_model.exam_assignment_vars[e_c, t_c])
                                    benders_cut_constant_term -= dual_val * x_bar_val
                        if benders_cut_variable_terms or abs(benders_cut_constant_term) > 1e-9: 
                            final_cut_rhs_expression = sum(benders_cut_variable_terms) + benders_cut_constant_term
                            if p_tuple_cut in benders_master_model.STUDENT_PATTERNS_FOR_OPT: 
                                try:
                                    benders_master_model.BendersCuts.add(
                                        (current_benders_iteration, p_tuple_cut), 
                                        expr=benders_master_model.student_pattern_expected_cost_vars[p_tuple_cut] >= final_cut_rhs_expression
                                    )
                                    cut_count_this_iter += 1
                                except Exception as add_cut_err: 
                                    logging.error(f"Error adding cut for pattern {p_tuple_cut}, iter {current_benders_iteration}: {add_cut_err}")
                    if cut_count_this_iter == 0 and current_benders_iteration < config.MAX_BENDERS_ITERATIONS: 
                        logging.warning("Warn: No valid cuts added this iteration.")
                    elif cut_count_this_iter > 0: logging.info(f"Added {cut_count_this_iter} cuts.")
                elif any_subproblem_failed_this_iteration: 
                    logging.warning("Skipping cut generation due to SP failure(s).")
                if current_benders_iteration >= config.MAX_BENDERS_ITERATIONS: 
                    logging.info("Max Benders iterations reached.")
        # MODIFICATION END
        
        # --- Final Summary and Reporting ---
        # ... (This section is unchanged) ...
        logging.info("\n>>> === Benders Decomposition for Patterns Finished === <<<")
        logging.info(f"Total Script Runtime: {time.time() - script_start_time:.2f}s")
        logging.info(f"Final LB (for OPTIMIZED patterns subset): {lower_bound:.4f}")
        logging.info(f"Final UB (for OPTIMIZED patterns subset): {upper_bound:.4f}" if upper_bound != float('inf') else "Final UB: inf")
        if config.CALCULATE_BEFORE_AFTER_OBJECTIVES:
            total_cost_all_target_patterns_after_optimization = 0.0
            if best_master_schedule_solution:
                logging.info("\nCalculating 'after' costs for ALL target patterns using the best global schedule...")
                canonical_costs_for_current_k = precomputed_canonical_schedule_costs_by_k_exams.get(k_for_canonical_generation)
                if canonical_costs_for_current_k is None and k_for_canonical_generation > 0:
                     logging.error(f"Canonical costs for k={k_for_canonical_generation} not found for 'after' calculation. Skipping.")
                else:
                    for p_tuple_final in target_student_pattern_list:
                        per_instance_cost_final = compute_expected_cost_for_student_pattern(p_tuple_final, all_time_slots, best_master_schedule_solution, config.SUBPROBLEM_SOLVER, k_for_canonical_generation, canonical_costs_for_current_k, print_details=config.PRINT_SUBPROBLEM_DETAILS_PER_ITERATION)
                        actual_student_count_final = target_student_pattern_counts[p_tuple_final]
                        total_cost_for_pattern_group_final = per_instance_cost_final * actual_student_count_final
                        if p_tuple_final in student_pattern_cost_tracking_dict:
                            student_pattern_cost_tracking_dict[p_tuple_final]['optimized_total_cost'] = total_cost_for_pattern_group_final
                        else:
                            student_pattern_cost_tracking_dict[p_tuple_final] = {'initial_total_cost': float('nan'), 'optimized_total_cost': total_cost_for_pattern_group_final, 'student_count': actual_student_count_final, 'is_in_optimization_subset': (p_tuple_final in student_patterns_for_benders_optimization)}
                        if per_instance_cost_final != float('inf'):
                            total_cost_all_target_patterns_after_optimization += total_cost_for_pattern_group_final
                        else:
                            total_cost_all_target_patterns_after_optimization = float('inf')
                            break
                    logging.info(f"Overall 'after' objective for ALL {len(target_student_pattern_list)} target patterns: {total_cost_all_target_patterns_after_optimization:.4f}")
            else:
                logging.warning("No best_master_schedule_solution found. Cannot calculate 'after' costs for all patterns.")
                for p_tuple_final in target_student_pattern_list:
                     if p_tuple_final in student_pattern_cost_tracking_dict:
                        student_pattern_cost_tracking_dict[p_tuple_final]['optimized_total_cost'] = float('nan')
        else:
            logging.info("\nSkipping 'after' objective calculation as per configuration.")
            for p_tuple_final in target_student_pattern_list:
                 if p_tuple_final in student_pattern_cost_tracking_dict:
                    student_pattern_cost_tracking_dict[p_tuple_final]['optimized_total_cost'] = float('nan')
        logging.info("\n--- Summary: Costs for ALL Target Patterns ---")
        results_df_list_all = []
        for p_tuple_res, data_res in student_pattern_cost_tracking_dict.items():
            results_df_list_all.append({'Pattern': str(p_tuple_res), 'Student Count': data_res.get('student_count', 'N/A'), 'In Opt Subset': data_res.get('is_in_optimization_subset', False), 'Initial Total Cost': data_res.get('initial_total_cost', float('nan')), 'Optimized Total Cost': data_res.get('optimized_total_cost', float('nan'))})
        if results_df_list_all:
            results_summary_df_all = pd.DataFrame(results_df_list_all)
            results_summary_df_all = results_summary_df_all.sort_values(by=['In Opt Subset', 'Student Count'], ascending=[False, False])
            logging.info("Full Results Summary Table (see log file for complete table if wide):\n" + results_summary_df_all.to_string(index=False, float_format="%.2f"))
            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                summary_str = results_summary_df_all[['Pattern', 'Student Count', 'Initial Total Cost', 'Optimized Total Cost']].head(20).to_string(index=False, float_format="%.2f")
                logging.info("\n--- Summary Table (Key Columns - see log for full) ---\n" + summary_str)
        else:
            logging.info("No optimization results to display in summary table.")
        if best_master_schedule_solution:
            logging.info(f"\n--- Best Global Schedule Found (Iter {iteration_of_best_master_schedule}) ---")
            active_assignments = { (e,t): val for (e,t), val in best_master_schedule_solution.items() if abs(val) > 1e-6}
            if active_assignments:
                schedule_df_list = [{'Exam': e, 'TimeSlot': t, 'Value': round(val,4)} for (e,t),val in active_assignments.items()]
                schedule_df_final = pd.DataFrame(schedule_df_list).sort_values(by=['TimeSlot', 'Exam'])
                logging.info("Best Schedule Assignments:\n" + schedule_df_final.to_string(index=False))
            else:
                logging.info("No active assignments (exam-to-slot with value > 0) in the best global schedule.")
        else:
            logging.info("\nNo best global schedule recorded (best_master_schedule_solution is None).")

    except ApplicationError as app_err: logging.critical(f"\nA Pyomo ApplicationError: {app_err}\n{traceback.format_exc()}")
    except ValueError as val_err: logging.critical(f"\nA ValueError: {val_err}\n{traceback.format_exc()}")
    except RuntimeError as run_err: logging.critical(f"\nA RuntimeError: {run_err}\n{traceback.format_exc()}")
    except Exception as e: logging.critical(f"\nAn critical error occurred: {type(e).__name__} - {e}\n{traceback.format_exc()}")
    finally:
        if persistent_master_solver and hasattr(persistent_master_solver, 'close'): 
            try: persistent_master_solver.close(); logging.info("Persistent MP solver closed.")
            except Exception as e_close_mp: logging.error(f"Error closing persistent MP solver: {e_close_mp}")
        
        calculate_penalty_for_gap.cache_clear()
        generate_canonical_slot_permutations.cache_clear()
        calculate_cost_for_chronological_schedule.cache_clear()
        logging.info("\nLRU Caches cleared.")
        logging.info("\n>>> === Script Finished === <<<")
        logging.shutdown()