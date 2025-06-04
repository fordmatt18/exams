"""
Functions for building the Pyomo models for the Benders decomposition:
- Master Problem (MP)
- Subproblem (SP) for calculating minimum expected student cost.
"""
import pyomo.environ as pyo
import logging # Import logging

def build_benders_master_problem(
    all_exams_set,
    all_time_slots_set,
    student_patterns_for_optimization_set, # Patterns for theta variables
    student_pattern_enrollment_counts_dict,
    reweight_subproblem_objective_by_pattern_count_flag,
    relax_master_variables_flag,
    allow_intra_pattern_conflicts_config_flag, # This is config.ALLOW_INTRA_PATTERN_CONFLICTS
    # New argument: A comprehensive list of patterns for which to enforce no-clash
    # This could be the same as student_patterns_for_optimization_set or a broader list
    # For example, all unique patterns from the input data that contain exams in all_exams_set.
    all_relevant_student_patterns_for_clash_constraints # List of tuples, e.g., [('E1','E2'), ('E1','E3','E4'), ...]
):
    """
    Builds the Benders Master Problem (MP).
    The MP decides exam assignments (x_et) and estimates student pattern costs (theta_s).

    Args:
        all_exams_set (list/set): Set of all unique exam IDs.
        all_time_slots_set (list/set): Set of all available time slot indices.
        student_patterns_for_optimization_set (list of tuples): List of student patterns
            (tuples of sorted exam IDs) for which theta variables are created.
        student_pattern_enrollment_counts_dict (dict): Maps student_pattern_tuple (from the set above)
            to its count of occurrences.
        reweight_subproblem_objective_by_pattern_count_flag (bool): See config.py.
        relax_master_variables_flag (bool): If True, x_et are continuous [0,1].
        allow_intra_pattern_conflicts_config_flag (bool): If True, student schedules (pi_s)
            can contain conflicts, penalized in U_pi. If False, a hard clash constraint
            is added to this MP.
        all_relevant_student_patterns_for_clash_constraints (list of tuples):
            A list of all unique student patterns (exam tuples) for which the
            no-clash constraint (sum_{e in E_s} x_et <= 1) should be enforced if
            allow_intra_pattern_conflicts_config_flag is False.
    Returns:
        pyomo.ConcreteModel: The constructed Benders Master Problem.
    """
    logging.info("--- Building Benders Master Problem (MP) ---") # Use logging
    model = pyo.ConcreteModel(name="Benders_Master_Problem")

    # --- Sets ---
    model.EXAMS = pyo.Set(initialize=all_exams_set)
    model.TIME_SLOTS = pyo.Set(initialize=all_time_slots_set)
    model.STUDENT_PATTERNS_FOR_OPT = pyo.Set(initialize=student_patterns_for_optimization_set)

    # A new set for indexing the clash constraints, if needed.
    # We need to ensure these patterns are tuples for Pyomo Set initialization.
    # And that the patterns themselves are hashable (tuples of strings are fine).
    # We also need a way to iterate through these patterns for the constraint rule.
    # Let's use a simple list of indices for the constraint if patterns are complex to index directly.
    model.ALL_RELEVANT_PATTERNS_FOR_CLASH = pyo.Set(initialize=range(len(all_relevant_student_patterns_for_clash_constraints)))


    # --- Variables ---
    if relax_master_variables_flag:
        model.exam_assignment_vars = pyo.Var(
            model.EXAMS, model.TIME_SLOTS, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0)
        )
    else:
        model.exam_assignment_vars = pyo.Var(
            model.EXAMS, model.TIME_SLOTS, domain=pyo.Binary
        )

    model.student_pattern_expected_cost_vars = pyo.Var(
        model.STUDENT_PATTERNS_FOR_OPT, domain=pyo.NonNegativeReals, initialize=0.0
    )

    # --- Constraints ---
    def assign_one_slot_per_exam_rule(m, exam_id):
        return sum(m.exam_assignment_vars[exam_id, ts_id] for ts_id in m.TIME_SLOTS) == 1
    model.AssignOneSlotPerExam = pyo.Constraint(model.EXAMS, rule=assign_one_slot_per_exam_rule)

    # Add student clash constraint if intra-pattern conflicts are NOT allowed
    if not allow_intra_pattern_conflicts_config_flag:
        logging.info(">>> MP: Adding hard constraints to prevent intra-pattern conflicts (sum_{e in E_s} x_et <= 1).")

        # The constraint is for each (student_pattern, time_slot)
        # student_pattern is a tuple of exam IDs
        def student_pattern_no_clash_rule(m, pattern_idx, ts_id):
            # Get the actual exam tuple for this pattern_idx
            student_pattern_exam_tuple = all_relevant_student_patterns_for_clash_constraints[pattern_idx]
            
            # Sum x_et for exams *in this specific student_pattern_exam_tuple* and *present in model.EXAMS*
            # This ensures we only consider exams that are part of the overall problem instance.
            # This also handles cases where a pattern might contain an exam not in all_exams_set (though ideally data is clean).
            sum_expr = sum(m.exam_assignment_vars[exam_id, ts_id]
                           for exam_id in student_pattern_exam_tuple
                           if exam_id in m.EXAMS) # Crucial check: exam_id must be in model.EXAMS
            return sum_expr <= 1

        model.StudentPatternNoClash = pyo.Constraint(
            model.ALL_RELEVANT_PATTERNS_FOR_CLASH, # Iterate over indices of the patterns
            model.TIME_SLOTS,
            rule=student_pattern_no_clash_rule
        )
    else:
        logging.info(">>> MP: Intra-pattern conflicts are allowed and penalized in subproblem costs (U_pi). No hard clash constraint in MP.")

    model.BendersCuts = pyo.Constraint(pyo.Any, pyo.Any)

    if reweight_subproblem_objective_by_pattern_count_flag:
        logging.info(">>> MP Objective: Sum of student_pattern_expected_cost_vars (theta_s represents total cost for pattern group).")
        obj_expr = sum(model.student_pattern_expected_cost_vars[p_tuple]
                       for p_tuple in model.STUDENT_PATTERNS_FOR_OPT)
    else:
        logging.info(">>> MP Objective: Weighted sum of student_pattern_expected_cost_vars (theta_s represents per-instance pattern cost).")
        obj_expr = sum(model.student_pattern_expected_cost_vars[p_tuple] *
                       student_pattern_enrollment_counts_dict.get(p_tuple, 1) # Default to 1 if somehow not in dict
                       for p_tuple in model.STUDENT_PATTERNS_FOR_OPT)

    model.Objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    logging.info("Benders Master Problem built successfully.")
    return model

def build_student_pattern_subproblem_model(
    student_pattern_exams_list,       # E_s: list of exams for this specific student pattern
    all_time_slots_set,               # T: global set of time slots
    master_schedule_solution_dict,    # x_et^(k): solution from MP for x_et
    personal_schedule_to_cost_map_for_pattern, # {pi_s: U_pi_s} for this pattern
    student_pattern_id_str="student_pattern", # Identifier for model naming
    subproblem_objective_weight=1.0   # Weight for this SP's objective (e.g., student count if reweighting)
):
    """
    Builds the Subproblem (SP) for a single student pattern.
    The SP calculates the minimum expected cost for this student pattern, given a
    fixed master schedule x_et^(k). (Corresponds to SP in Sec 2.3.3 of LaTeX)

    Args:
        student_pattern_exams_list (list): Exams E_s for this student pattern.
        all_time_slots_set (list/set): Global set of time slots T.
        master_schedule_solution_dict (dict): Current MP solution x_et^(k).
                                              Maps (exam_id, ts_id) to its value.
        personal_schedule_to_cost_map_for_pattern (dict): Maps a personal schedule pi_s
            (tuple of (exam_id, ts_id) assignments for exams in E_s) to its cost U_pi_s.
        student_pattern_id_str (str): A string identifier for this pattern, for model naming.
        subproblem_objective_weight (float): Weight applied to this SP's objective.
                                             Usually 1.0, or student_count if reweighting in SP.

    Returns:
        pyomo.ConcreteModel: The constructed subproblem model. Returns None if
                             personal_schedule_to_cost_map_for_pattern is empty.
    """
    if not personal_schedule_to_cost_map_for_pattern:
        # This can happen if k_exams > num_slots and allow_slot_reuse is False for canonical generation.
        return None

    model = pyo.ConcreteModel(name=f"Subproblem_MinExpectedCost_{student_pattern_id_str}")

    # --- Sets ---
    # Pi_s: Set of all possible personal schedules (pi) for this student pattern.
    # These are the keys of personal_schedule_to_cost_map_for_pattern.
    # Each personal_schedule is a tuple of (exam_id, ts_id) items.
    set_of_personal_schedules = list(personal_schedule_to_cost_map_for_pattern.keys())
    # We use indices to refer to these personal schedules in Pyomo.
    personal_schedule_indices = list(range(len(set_of_personal_schedules)))

    model.EXAMS_IN_PATTERN = pyo.Set(initialize=student_pattern_exams_list) # E_s
    model.TIME_SLOTS = pyo.Set(initialize=all_time_slots_set)               # T
    model.PERSONAL_SCHEDULE_INDICES = pyo.Set(initialize=personal_schedule_indices) # Index for Pi_s

    # --- Parameters ---
    # U_pi_s: Cost of each personal schedule pi_s for this student pattern.
    model.cost_of_personal_schedule_param = pyo.Param(
        model.PERSONAL_SCHEDULE_INDICES,
        initialize={idx: personal_schedule_to_cost_map_for_pattern[set_of_personal_schedules[idx]]
                    for idx in model.PERSONAL_SCHEDULE_INDICES}
    )

    # x_et^(k): Fixed master schedule solution values for exams in this pattern.
    def master_schedule_param_init(m, exam_id, ts_id):
        return master_schedule_solution_dict.get((exam_id, ts_id), 0.0) # Default to 0 if exam not in x_bar (should not happen for E_s)
    model.master_schedule_param = pyo.Param(
        model.EXAMS_IN_PATTERN, model.TIME_SLOTS,
        initialize=master_schedule_param_init,
        mutable=True # Though fixed per SP solve, mutable if model is reused with new x_bar
    )

    # --- Variables ---
    # y_s,pi: Probability or weight assigned to personal schedule pi for this student pattern s.
    model.personal_schedule_prob_vars = pyo.Var(
        model.PERSONAL_SCHEDULE_INDICES, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0)
    )

    # --- Objective Function (Eq. \ref{eq:sp_obj_new}) ---
    # Minimize sum_{pi in Pi_s} (y_s,pi * U_pi_s)
    model.Objective = pyo.Objective(
        expr=subproblem_objective_weight * sum(
            model.personal_schedule_prob_vars[idx] * model.cost_of_personal_schedule_param[idx]
            for idx in model.PERSONAL_SCHEDULE_INDICES
        ),
        sense=pyo.minimize
    )

    # --- Constraints ---
    # Linking Constraint (Eq. \ref{eq:sp_link_new}): sum_{pi in Pi_s,e,t} y_s,pi <= x_et^(k)
    # Dual variable for this constraint is lambda_s,e,t^(k).
    model.LinkingConstraint = pyo.Constraint(model.EXAMS_IN_PATTERN, model.TIME_SLOTS)

    # Pre-calculate which personal schedules use which (exam, slot) pair for efficiency.
    # Map (exam_id, ts_id) to a list of personal_schedule_indices that assign that exam to that slot.
    personal_schedules_using_exam_at_slot = {
        (exam_id, ts_id): []
        for exam_id in model.EXAMS_IN_PATTERN for ts_id in model.TIME_SLOTS
    }
    for idx in model.PERSONAL_SCHEDULE_INDICES:
        personal_schedule_tuple = set_of_personal_schedules[idx] # e.g., ( (E1,T0), (E2,T1) )
        for exam_in_schedule, slot_in_schedule in personal_schedule_tuple:
            # Ensure the exam from the schedule tuple is one of the exams this SP is for.
            if exam_in_schedule in model.EXAMS_IN_PATTERN and slot_in_schedule in model.TIME_SLOTS:
                personal_schedules_using_exam_at_slot[(exam_in_schedule, slot_in_schedule)].append(idx)

    for exam_id_constr in model.EXAMS_IN_PATTERN:
        for ts_id_constr in model.TIME_SLOTS:
            # Get all y_s,pi variables where pi assigns exam_id_constr to ts_id_constr.
            relevant_schedule_indices = personal_schedules_using_exam_at_slot.get((exam_id_constr, ts_id_constr), [])
            sum_of_probs_for_exam_at_slot = sum(model.personal_schedule_prob_vars[idx]
                                                for idx in relevant_schedule_indices)
            model.LinkingConstraint.add(
                (exam_id_constr, ts_id_constr),
                expr=sum_of_probs_for_exam_at_slot <= model.master_schedule_param[exam_id_constr, ts_id_constr]
            )

    # Normalization Constraint (Eq. \ref{eq:sp_norm_new}): sum_{pi in Pi_s} y_s,pi = 1
    # Dual variable for this constraint is nu_s^(k).
    model.NormalizationConstraint = pyo.Constraint(
        expr=sum(model.personal_schedule_prob_vars[idx] for idx in model.PERSONAL_SCHEDULE_INDICES) == 1
    )

    # Suffix for importing dual variable values.
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model