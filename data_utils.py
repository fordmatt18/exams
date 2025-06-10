"""
Utilities for generating simulated exam enrollment data and loading/structuring
enrollments from CSV files.
"""
import pandas as pd
import random
import ast # For safely evaluating string representations of lists
import logging

def generate_simulated_enrollment_data(num_exams, num_students, num_slots, exams_per_student):
    """
    Generates simulated exam enrollment data.

    Not typically used if loading data from CSV, but useful for testing or
    when real data is unavailable.

    Args:
        num_exams (int): Total number of unique exams to generate.
        num_students (int): Total number of students to generate.
        num_slots (int): Total number of available time slots.
        exams_per_student (int): The fixed number of exams each student is enrolled in.

    Returns:
        tuple: Contains:
            - Exams_list (list): List of exam IDs (e.g., ['E1', 'E2', ...]).
            - TimeSlots_list (list): List of time slot indices (e.g., [0, 1, ...]).
            - Students_list (list): List of student IDs (e.g., ['S1', 'S2', ...]).
            - Enrollments_dict (dict): Maps student ID to a list of exam IDs they take.
            - ConflictingPairs_list (list): List of exam pairs that share at least one student.
            - N_common_dict (dict): Maps exam pairs to the number of students taking both.
    """
    logging.info("--- Generating Simulated Data ---")
    if num_exams <= 0 or num_students <= 0 or num_slots <= 0:
         raise ValueError("Number of exams, students, and slots must be positive.")
    if not isinstance(exams_per_student, int) or not (1 <= exams_per_student <= num_exams):
         raise ValueError(f"exams_per_student ({exams_per_student}) must be an integer between 1 and num_exams ({num_exams}).")

    Exams_list = [f'E{i+1}' for i in range(num_exams)]
    Students_list = [f'S{i+1}' for i in range(num_students)]
    TimeSlots_list = list(range(num_slots)) # Corresponds to T in LaTeX
    Enrollments_dict = {s: [] for s in Students_list} # Corresponds to E_s for each s
    # ExamEnrollmentCount = {e: 0 for e in Exams_list} # Not directly returned, but used internally

    logging.info(f"Assigning exactly {exams_per_student} exams to each student.")
    for s in Students_list:
        # Ensure we don't try to sample more exams than available
        num_to_sample = min(exams_per_student, len(Exams_list))
        if num_to_sample > 0 and Exams_list: # Check if there are exams to sample from
            exams_taken_by_student = random.sample(Exams_list, k=num_to_sample)
        else:
            exams_taken_by_student = []
        Enrollments_dict[s] = exams_taken_by_student
        # for e in exams_taken_by_student:
        #     if e not in ExamEnrollmentCount: ExamEnrollmentCount[e] = 0
        #     ExamEnrollmentCount[e] += 1

    N_common_dict = {} # Number of common students for pairs of exams
    ConflictingPairs_list = [] # List of (exam1, exam2) tuples that have common students
    for i, e1 in enumerate(Exams_list):
        for j, e2 in enumerate(Exams_list):
            if i < j: # Avoid duplicate pairs and self-pairs
                common_students_count = 0
                for s_check in Students_list:
                    student_exams = Enrollments_dict.get(s_check, [])
                    if e1 in student_exams and e2 in student_exams:
                        common_students_count += 1
                if common_students_count > 0:
                    pair = tuple(sorted((e1, e2))) # Canonical representation of the pair
                    N_common_dict[pair] = common_students_count
                    ConflictingPairs_list.append(pair)

    logging.info(f"Generated {num_exams} exams, {num_students} students, {num_slots} time slots.")
    logging.info(f"Total student enrollments: {sum(len(v) for v in Enrollments_dict.values())}")
    ConflictingPairs_list = sorted(list(set(ConflictingPairs_list))) # Ensure uniqueness and sort
    logging.info(f"Number of unique conflicting exam pairs: {len(ConflictingPairs_list)}")
    if len(ConflictingPairs_list) == 0 and num_exams > 1 and num_students > 0 and exams_per_student > 1:
        logging.warning("No conflicting exam pairs found. This might be unusual depending on parameters.")
    logging.info("-" * 30)
    return Exams_list, TimeSlots_list, Students_list, Enrollments_dict, ConflictingPairs_list, N_common_dict

def load_and_structure_student_enrollments(csv_file_path):
    """
    Loads student enrollment data from a CSV file and structures it.

    The CSV file must contain 'student' and 'exams' columns. The 'exams' column
    should contain a string representation of a list of exam IDs for each student.

    Args:
        csv_file_path (str): Path to the enrollment CSV file.

    Returns:
        tuple: Contains:
            - enrollments_grouped_by_pattern_size (dict):
                Maps number of exams (int) to another dict. This inner dict maps
                a student_pattern_tuple (tuple of sorted exam IDs) to its count (int).
                Example: {3: {('E1','E2','E3'): 10, ('E1','E2','E4'): 5}, 4: ...}
            - unique_exam_ids (list): Sorted list of all unique exam IDs found in the data (E in LaTeX).
            - max_exams_per_student (int): Maximum number of exams any single student is enrolled in.
        Returns (None, None, 0) if loading fails or data is invalid.
    """
    logging.info(f"--- Loading and Structuring Student Enrollments from {csv_file_path} ---")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        logging.error(f"ERROR: CSV file not found at {csv_file_path}")
        return None, None, 0
    if 'student' not in df.columns or 'exams' not in df.columns:
        logging.error("ERROR: CSV must contain 'student' and 'exams' columns.")
        return None, None, 0

    enrollments_grouped_by_pattern_size = {} # Key: num_exams_in_pattern, Value: {pattern_tuple: count}
    unique_exam_ids = set()
    max_exams_per_student = 0

    for _, row in df.iterrows():
        student_id = row['student']
        exams_list_str = row['exams']
        try:
            # Safely evaluate the string representation of the list
            exams_list_for_student = ast.literal_eval(exams_list_str)
            if not isinstance(exams_list_for_student, list):
                logging.warning(f"Warning: 'exams' column for student {student_id} is not a list: {exams_list_str}")
                continue

            # Standardize exam IDs (string, stripped) and sort to create a canonical pattern
            exams_in_current_pattern = sorted([str(exam_id).strip() for exam_id in exams_list_for_student])
            unique_exam_ids.update(exams_in_current_pattern)

            current_pattern_size = len(exams_in_current_pattern)
            max_exams_per_student = max(max_exams_per_student, current_pattern_size)
            student_pattern_tuple = tuple(exams_in_current_pattern) # This is E_s for a student

            # Group by pattern size
            if current_pattern_size not in enrollments_grouped_by_pattern_size:
                enrollments_grouped_by_pattern_size[current_pattern_size] = {}

            # Count occurrences of each unique student pattern
            enrollments_grouped_by_pattern_size[current_pattern_size][student_pattern_tuple] = \
                enrollments_grouped_by_pattern_size[current_pattern_size].get(student_pattern_tuple, 0) + 1

        except (ValueError, SyntaxError) as e:
            logging.warning(f"Warning: Could not parse exams for student {student_id}: '{exams_list_str}'. Error: {e}")
            continue

    if not unique_exam_ids:
        logging.warning("Warning: No valid exam enrollments found in CSV.")
        return None, None, 0

    logging.info(f"Loaded data. Found {len(unique_exam_ids)} unique exams across all students.")
    logging.info(f"Max exams any student takes: {max_exams_per_student}")
    logging.info("-" * 30)
    return enrollments_grouped_by_pattern_size, sorted(list(unique_exam_ids)), max_exams_per_student