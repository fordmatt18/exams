"""
Configuration parameters for the Benders decomposition-based exam timetabling optimizer.
"""

# --- Problem Instance Parameters ---
STUDENT_ENROLLMENT_CSV_PATH = "sp24_by_student.csv"
"""Path to the CSV file containing student enrollment data.
CSV must have 'student' and 'exams' columns.
'exams' column should be a string representation of a list of exam IDs.
"""

NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION = 4
"""The specific number of exams in a student's schedule (pattern)
that the Benders decomposition will focus on optimizing.
"""

STUDENT_PATTERN_SUBSET_FRACTION = 0.1
"""Fraction of unique student patterns (with NUM_EXAMS_IN_PATTERNS_FOR_OPTIMIZATION exams)
to be included in the Benders decomposition master problem.
A value of 1.0 means all such patterns are included.
"""

NUM_TIME_SLOTS = 24
"""Total number of available time slots for scheduling exams (T in LaTeX)."""

ALLOW_INTRA_PATTERN_CONFLICTS = True # consider removing, what is use case of it being false
"""
Recommended Setting: True, this ensures feasibility and may make schedules better overall at the cost of a few conflicts.
If True, a student's personal schedule (pi_s) can have multiple exams in the same slot,
and this is penalized by p0. The MP will not have hard student-clash constraints.
If False, pi_s must be conflict-free, and the MP includes hard student-clash constraints.
Corresponds to 'allow_conflicts' in LaTeX.
"""

GAP_PENALTY_DICT = {
    # Key: Gap Size, Value: Penalty
    0: 100,    # Penalty for a zero gap (conflict)
    1: 9,
    2: 8.75,
    3: 7.25,
    4: 6.25,
    5: 5.75,
    6: 3,
    # Sentinel key for large gaps. Any gap >= 7 will use this penalty.
    # The key (100) is arbitrary; it just needs to be a key that won't
    # conflict with an actual gap size.
    100: 2
}
"""
A dictionary defining the cost structure for student schedules based on gap sizes.
- Key `0` is the penalty for a direct conflict (zero gap).
- Keys `1, 2, ...` are penalties for specific small gaps.
- The largest key (sentinel `100`) defines the penalty for any gap size greater
  than the largest specific gap defined (e.g., for all gaps >= 7 in this config).
"""

# --- Benders Algorithm Parameters ---
MAX_BENDERS_ITERATIONS = 5
"""
Recommended Setting: Large number so it doesn't cap out (10000)
Maximum number of iterations for the Benders decomposition algorithm (k_max in LaTeX)."""

BENDERS_ABSOLUTE_TOLERANCE = 1e-4
"""Absolute tolerance for convergence (UB - LB) (epsilon_abs in LaTeX)."""

BENDERS_RELATIVE_TOLERANCE = 1e-4
"""Relative tolerance for convergence ( (UB - LB) / |UB| ) (epsilon_rel in LaTeX)."""

RELAX_MASTER_PROBLEM_VARIABLES = False
"""
Recommended Setting: False, only run true for algorithm experimentation/development.
If True, master problem scheduling variables (x_et) are continuous [0,1].
If False, they are binary {0,1} (relax_mp in LaTeX).
"""

REWEIGHT_SUBPROBLEM_OBJECTIVE_BY_PATTERN_COUNT = False
"""If True, the subproblem objective for a pattern is weighted by the number of students
sharing that pattern. This means theta_s in MP represents the total cost for that group.
If False, theta_s represents the per-instance cost for a pattern, and the MP objective
weights theta_s by the student count.
"""

# --- Solver Configuration ---
MASTER_PROBLEM_SOLVER = 'gurobi'
"""Name of the solver to be used for the master problem."""

SUBPROBLEM_SOLVER = 'gurobi'
"""Name of the solver to be used for the subproblems."""

RANDOM_SEED = 47
"""Seed for random number generation, ensuring reproducibility."""

# --- Execution Control ---
USE_PARALLEL_SUBPROBLEM_SOLVING = True
"""If True, subproblems will be solved in parallel using multiple processes."""

NUM_SUBPROBLEM_WORKERS = None
"""Number of worker processes for parallel subproblem solving.
None means use os.cpu_count().
"""

USE_PERSISTENT_MASTER_PROBLEM_SOLVER = False
"""If True, attempts to use a persistent solver for the master problem,
which can be faster for iterative solves by avoiding model rebuild overhead.
"""

# --- Debug and Print Flags ---
PRINT_MASTER_PROBLEM_DEBUG_PERSISTENT = False
"""If True and using a persistent MP solver, prints detailed debug info from the solver utility."""

PRINT_MASTER_PROBLEM_DETAILS_PER_ITERATION = False
"""If True, prints details of the master problem solution and objective at each iteration."""

PRINT_SUBPROBLEM_DETAILS_PER_ITERATION = False
"""If True, prints details during the solving of individual subproblems (passed to worker task)."""

PRINT_MASTER_SOLUTION_X_BAR_PER_ITERATION = False
"""If True, prints the non-zero values of the master problem's x_et solution (x_bar) at each iteration."""

PRINT_SUBPROBLEM_DUALS_PER_ITERATION = False
"""If True, prints details of the dual variables obtained from subproblems, used for cut generation."""

CALCULATE_BEFORE_AFTER_OBJECTIVES = False
"""If True, calculates the total objective for all target student patterns
using an initial schedule (before optimization) and the best found schedule
(after optimization). This can be time-consuming for large datasets.
"""

LOG_FILE_NAME = "benders_optimization.log"
"""Name of the file to save detailed logs."""

LOG_LEVEL_CONSOLE = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
"""Logging level for console output."""

LOG_LEVEL_FILE = "DEBUG"
"""Logging level for file output."""