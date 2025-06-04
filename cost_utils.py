"""
Utilities for calculating costs associated with exam schedules, including
penalties for exam spacing and generation of canonical schedule patterns.
"""
import itertools
import math
from functools import lru_cache

def _make_penalty_params_hashable_for_cache(penalty_params_dict):
    """
    Converts mutable parts of penalty_params_dict (set 'D', dict 'p_d')
    to their immutable, hashable counterparts (frozenset, sorted tuple of items)
    for use with lru_cache.

    Args:
        penalty_params_dict (dict): The penalty parameters dictionary.

    Returns:
        tuple: A tuple of key-value pairs, where 'D' and 'p_d' values are hashable.
    """
    # Ensure 'D' and 'p_d' exist, even if empty, for consistent tuple structure
    d_val = frozenset(penalty_params_dict.get('D', set()))
    p_d_val = tuple(sorted(penalty_params_dict.get('p_d', {}).items()))
    # Other params are typically numbers, already hashable
    p_mplus_val = penalty_params_dict.get('p_Mplus', 0)
    p0_val = penalty_params_dict.get('p0', 0)
    return (('D', d_val), ('p_d', p_d_val), ('p_Mplus', p_mplus_val), ('p0', p0_val))

@lru_cache(maxsize=None) # Cache results as penalties are frequently recalculated for same gaps
def calculate_penalty_for_gap(gap, penalty_parameters_tuple):
    """
    Calculates the penalty for a given gap size between two consecutive exams.
    Corresponds to P(g) in LaTeX (Eq. \ref{eq:penalty_func}).

    Args:
        gap (int or float): The time difference between two exams.
        penalty_parameters_tuple (tuple): Hashable tuple of penalty parameters,
                                          as returned by _make_penalty_params_hashable_for_cache.

    Returns:
        float: The calculated penalty for the gap.
    """
    # Reconstruct parameters from the hashable tuple
    local_penalty_params = {}
    D_set = frozenset()
    p_d_dict = {}
    for key, value in penalty_parameters_tuple:
        if key == 'D': D_set = value
        elif key == 'p_d': p_d_dict = dict(value) # Convert back to dict for easy lookup
        else: local_penalty_params[key] = value

    M_threshold = max(D_set) if D_set else -1 # M = max D in LaTeX
    penalty_for_gap_ge_M = local_penalty_params.get('p_Mplus', 0) # p_M+ in LaTeX
    penalty_for_zero_gap = local_penalty_params.get('p0', 0) # p_0 in LaTeX

    if abs(gap - 0) < 1e-9: # Using tolerance for float comparison
        return penalty_for_zero_gap
    elif gap < 0: # Should not occur with time-sorted slots
        return 0
    elif gap in D_set:
        return p_d_dict.get(gap, 0) # p_d in LaTeX
    elif M_threshold >= 0 and gap >= M_threshold: # Check M_threshold >= 0 before using it
        return penalty_for_gap_ge_M
    else: # Gap is > 0, not in D, and < M (or D is empty)
        return 0

@lru_cache(maxsize=128) # Cache results for common (k, T, allow_reuse) combinations
def generate_canonical_slot_permutations(num_exams_in_pattern, time_slots_tuple, allow_slot_reuse_in_pattern):
    """
    Generates all unique, canonical slot assignment patterns for a given number of exams
    and available time slots. "Canonical" means for placeholder exams (e.g., Exam1, Exam2,...).
    The order of slots in the output tuple corresponds to the order of placeholder exams.

    Args:
        num_exams_in_pattern (int): The number of exams (k) to assign slots to.
        time_slots_tuple (tuple): Tuple of available time slot indices.
        allow_slot_reuse_in_pattern (bool): If True, multiple exams in the pattern can be
                                            assigned to the same slot (itertools.product).
                                            If False, each exam must have a unique slot
                                            (itertools.permutations).

    Returns:
        tuple: A tuple of tuples, where each inner tuple is a slot assignment pattern
               (e.g., ((slot1_for_exam1, slot2_for_exam2), (slotA_for_exam1, slotB_for_exam2), ...)).
               Returns None if generation encounters a critical error (e.g., too many patterns).
               Returns an empty tuple if no patterns are possible (e.g., k > num_slots and no reuse).
    """
    num_available_slots = len(time_slots_tuple)
    slot_patterns = []

    if num_exams_in_pattern == 0:
        return tuple() # No exams, no slots to assign

    num_potential_patterns = 0
    if allow_slot_reuse_in_pattern:
        # Each of k exams can go into any of n_slots slots: n_slots^k possibilities
        try:
            num_potential_patterns = num_available_slots ** num_exams_in_pattern
        except OverflowError:
            print(f"    ERROR: Canonical pattern count overflow ({num_available_slots}^{num_exams_in_pattern}).")
            return None
        slot_iterator = itertools.product(time_slots_tuple, repeat=num_exams_in_pattern)
    else:
        # k exams must be placed in k distinct slots out of n_slots: P(n_slots, k) possibilities
        if num_exams_in_pattern > num_available_slots:
            return tuple() # Not enough unique slots for the exams
        try:
            if hasattr(math, 'perm'): # Python 3.8+
                num_potential_patterns = math.perm(num_available_slots, num_exams_in_pattern)
            else: # Fallback for older Python
                num_potential_patterns = math.factorial(num_available_slots) // math.factorial(num_available_slots - num_exams_in_pattern)
        except (OverflowError, ValueError): # ValueError if n_slots < k_exams (already handled) or negative factorial
            print(f"    ERROR: Canonical pattern permutation count error for P({num_available_slots}, {num_exams_in_pattern}).")
            return None
        slot_iterator = itertools.permutations(time_slots_tuple, num_exams_in_pattern)

    # Thresholds to prevent excessive memory/time usage
    warning_threshold = 1_000_000
    error_threshold = 10_000_000
    if num_potential_patterns > warning_threshold:
        print(f"    WARNING: Canonical slot pattern generation for {num_exams_in_pattern} exams, {num_available_slots} slots: "
              f"{num_potential_patterns:,} potential patterns!")
    if num_potential_patterns > error_threshold:
        print(f"    ERROR: Too many canonical patterns ({num_potential_patterns:,}) for {num_exams_in_pattern} exams. "
              f"Aborting for this k.")
        return None

    for slot_combination in slot_iterator:
        slot_patterns.append(slot_combination)

    return tuple(slot_patterns)

@lru_cache(maxsize=1024) # Cache costs for frequently occurring chronological slot patterns
def calculate_cost_for_chronological_schedule(chronological_slot_pattern_tuple, penalty_parameters_tuple):
    """
    Calculates the total cost (U_pi in LaTeX) for a personal schedule,
    given that the exams in that schedule are already sorted by their assigned time slots.
    The cost includes sum of gap penalties and the time of the last exam.

    Args:
        chronological_slot_pattern_tuple (tuple): A tuple of time slots, sorted chronologically,
                                                  representing the schedule for a student's exams.
                                                  E.g., (slot_for_exam_A, slot_for_exam_B, ...)
                                                  where exam_A occurs before or at the same time as exam_B.
        penalty_parameters_tuple (tuple): Hashable tuple of penalty parameters.

    Returns:
        float: The total cost U_pi for this specific chronological arrangement of slots.
    """
    num_exams_in_schedule = len(chronological_slot_pattern_tuple)
    if num_exams_in_schedule == 0:
        return 0.0

    current_total_gap_penalty = 0.0
    if num_exams_in_schedule >= 2:
        for k_idx in range(num_exams_in_schedule - 1):
            gap = chronological_slot_pattern_tuple[k_idx+1] - chronological_slot_pattern_tuple[k_idx]
            penalty_for_this_gap = calculate_penalty_for_gap(gap, penalty_parameters_tuple)
            current_total_gap_penalty += penalty_for_this_gap

    time_of_last_exam = chronological_slot_pattern_tuple[-1]
    # U_pi = TotalPenalty_pi + LastTime_pi (from LaTeX)
    personal_schedule_cost = current_total_gap_penalty + time_of_last_exam
    return personal_schedule_cost

def calculate_canonical_schedule_costs(
    canonical_placeholder_exams_list,
    time_slots_set,
    penalty_params_dict,
    allow_slot_reuse_within_canonical_schedule=False
):
    """
    Generates all canonical schedules for a set of placeholder exams and calculates
    the cost (U_pi) for each. A canonical schedule assigns each placeholder exam
    to a time slot.

    Args:
        canonical_placeholder_exams_list (list): A list of placeholder exam IDs
                                                 (e.g., ['__PEX__0', '__PEX__1']).
        time_slots_set (list or set): The set of available time slots (T in LaTeX).
        penalty_params_dict (dict): The dictionary of penalty parameters.
        allow_slot_reuse_within_canonical_schedule (bool): If True, placeholder exams
                                                           can be assigned to the same slot.

    Returns:
        dict: A dictionary mapping a canonical_schedule_tuple to its cost (U_pi).
              A canonical_schedule_tuple is like ((ph_exam1, slotA), (ph_exam2, slotB), ...).
              Returns None if canonical slot pattern generation fails.
              Returns an empty dict if no schedules are possible or num_exams is 0.
    """
    canonical_schedule_to_cost_map = {}
    num_exams_in_pattern = len(canonical_placeholder_exams_list)

    if num_exams_in_pattern == 0:
        return {}

    time_slots_tuple = tuple(sorted(list(time_slots_set))) # Ensure consistent order for caching
    penalty_parameters_tuple_for_cache = _make_penalty_params_hashable_for_cache(penalty_params_dict)

    # Generate permutations of slots for the placeholder exams
    # E.g., if num_exams=2, slots=(0,1), result could be ((0,1), (1,0), (0,0), (1,1))
    # where each inner tuple (s1,s2) means placeholder_exam1->s1, placeholder_exam2->s2
    canonical_slot_permutations = generate_canonical_slot_permutations(
        num_exams_in_pattern,
        time_slots_tuple,
        allow_slot_reuse_within_canonical_schedule
    )

    if canonical_slot_permutations is None: # Error during generation
        return None
    if not canonical_slot_permutations and num_exams_in_pattern > 0: # No patterns possible
        return {}

    # For each canonical slot permutation, form the (placeholder_exam, slot) assignment
    # and calculate its cost.
    for slot_assignment_for_placeholders in canonical_slot_permutations:
        current_canonical_schedule_assignment_list = []
        for i in range(num_exams_in_pattern):
            placeholder_exam = canonical_placeholder_exams_list[i]
            assigned_slot = slot_assignment_for_placeholders[i]
            current_canonical_schedule_assignment_list.append((placeholder_exam, assigned_slot))

        # This tuple represents one full assignment pi for the placeholder exams
        # e.g., (('PEX0', slot_A), ('PEX1', slot_B))
        canonical_schedule_assignment_tuple = tuple(current_canonical_schedule_assignment_list)

        # To calculate cost, sort this assignment by time
        schedule_sorted_by_time = sorted(current_canonical_schedule_assignment_list, key=lambda item: item[1])
        chronological_slots_only_tuple = tuple(item[1] for item in schedule_sorted_by_time)

        # Calculate U_pi for this time-ordered sequence of slots
        schedule_cost = calculate_cost_for_chronological_schedule(
            chronological_slots_only_tuple,
            penalty_parameters_tuple_for_cache
        )
        canonical_schedule_to_cost_map[canonical_schedule_assignment_tuple] = schedule_cost

    return canonical_schedule_to_cost_map