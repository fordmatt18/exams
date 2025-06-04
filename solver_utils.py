"""
Utility functions for solving Pyomo models, including handling for
standard and persistent solvers, and extracting results like duals.
"""
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

def solve_master_problem_persistent(
    persistent_solver_instance,
    model,
    suppress_solver_output=False,
    iteration_num=-1,
    print_debug_flag=False
):
    """
    Solves a Pyomo model using a pre-initialized persistent solver.

    Args:
        persistent_solver_instance: An initialized Pyomo persistent solver instance.
        model (pyomo.ConcreteModel): The Pyomo model to solve.
        suppress_solver_output (bool): If True, solver's console output is suppressed.
        iteration_num (int): Current Benders iteration number, for debug printing.
        print_debug_flag (bool): If True, prints detailed debug information.

    Returns:
        tuple: (results_object, termination_condition)
               results_object can be None if an error occurs.
               termination_condition is a Pyomo TerminationCondition enum.
    """
    results_object = None
    term_cond = TerminationCondition.error # Default to error

    try:
        # For persistent solvers, load_solutions=True is often desired to update the model instance.
        results_object = persistent_solver_instance.solve(model, load_solutions=True, tee=not suppress_solver_output)
        if print_debug_flag:
            print(f"DEBUG Iter {iteration_num} (Persistent MP Solve with load_solutions=True):")
            try:
                # Accessing lower_bound can sometimes be problematic depending on solver/status
                print(f"  Solver reported objective (from results.problem.lower_bound): {results_object.problem.lower_bound}")
            except Exception as e_res_obj:
                print(f"  Could not get objective from results.problem: {e_res_obj}")

            # Check theta variable values after solve
            if hasattr(model, 'student_pattern_expected_cost_vars'): # Check if theta exists
                for pattern_key in model.student_pattern_expected_cost_vars:
                    theta_var = model.student_pattern_expected_cost_vars[pattern_key]
                    if hasattr(theta_var, 'value') and theta_var.value is not None:
                        print(f"  model.student_pattern_expected_cost_vars[{pattern_key}].value: {theta_var.value}")
                    else:
                        print(f"  model.student_pattern_expected_cost_vars[{pattern_key}] value is None or not present.")
        if results_object and results_object.solver:
            term_cond = results_object.solver.termination_condition
    except Exception as e:
        print(f"Error solving with persistent MP solver: {e}")
        # term_cond remains TerminationCondition.error
    return results_object, term_cond

def solve_pyomo_model(model, solver_name='cbc', suppress_solver_output=False):
    """
    Solves a Pyomo model using a standard (non-persistent) solver.

    Args:
        model (pyomo.ConcreteModel): The Pyomo model to solve.
        solver_name (str): Name of the solver to use (e.g., 'cbc', 'gurobi').
        suppress_solver_output (bool): If True, solver's console output is suppressed.

    Returns:
        tuple: (results_object, termination_condition)
               results_object can be None if an error occurs.
               termination_condition is a Pyomo TerminationCondition enum.
    """
    solver_instance = None
    try:
        solver_instance = SolverFactory(solver_name)
    except Exception as e: # Covers ApplicationError if solver not found/configured
        print(f"Solver Configuration Error for '{solver_name}': {e}")
        return None, TerminationCondition.error

    if not solver_instance.available(exception_flag=False): # Check if solver is actually available
        print(f"Solver '{solver_name}' is not available or not found in PATH.")
        return None, TerminationCondition.solverFailure # Or a more specific condition

    results_object = None
    term_cond = TerminationCondition.error # Default to error
    try:
        results_object = solver_instance.solve(model, tee=not suppress_solver_output)
        if results_object and results_object.solver:
            term_cond = results_object.solver.termination_condition
    except Exception as e: # Broad exception for solver errors during the solve call
        model_name_str = model.name if model and hasattr(model, 'name') else 'UnnamedModel'
        print(f"Solver Error ({type(e).__name__}) with '{solver_name}' on model '{model_name_str}': {e}")
        # term_cond remains TerminationCondition.error
    return results_object, term_cond

def solve_subproblem_and_get_duals(
    subproblem_model,
    solver_name='cbc',
    suppress_solver_output=True
):
    """
    Solves a student pattern subproblem model and extracts its objective value and duals
    for the linking constraints.

    Args:
        subproblem_model (pyomo.ConcreteModel): The subproblem model to solve.
                                                Must have a Suffix named 'dual' and
                                                a Constraint named 'LinkingConstraint'.
        solver_name (str): Name of the solver.
        suppress_solver_output (bool): If True, suppress solver's console output.

    Returns:
        tuple: Contains:
            - results_object: The solver results object.
            - objective_value (float): Optimal objective value of the subproblem.
                                       float('inf') if infeasible or error.
            - linking_constraint_duals (dict): Maps (exam_id, ts_id) to dual value.
                                               Empty if infeasible or error.
            - solver_status (pyomo.opt.SolverStatus): Solver status enum.
            - termination_condition (pyomo.opt.TerminationCondition): Termination condition enum.
    """
    if subproblem_model is None:
        return None, float('inf'), {}, SolverStatus.error, TerminationCondition.error

    results_object, term_cond = solve_pyomo_model(
        subproblem_model, solver_name=solver_name, suppress_solver_output=suppress_solver_output
    )

    objective_value = float('inf') # Default for non-optimal or error
    linking_constraint_duals = {}
    solver_status = SolverStatus.error # Default
    if results_object and results_object.solver:
        solver_status = results_object.solver.status

    # Check if the solve was successful enough to extract values
    # TerminationCondition.optimal is ideal.
    # TerminationCondition.feasible might occur for LPs if optimal not reached due to limits,
    # but for SPs, we typically expect optimal.
    solved_ok_for_duals = term_cond == TerminationCondition.optimal

    if solved_ok_for_duals:
        try:
            objective_value = pyo.value(subproblem_model.Objective, exception=False)
            if objective_value is None: # Should not happen if optimal, but as a safeguard
                objective_value = float('inf')
        except Exception: # Should not happen if optimal
            objective_value = float('inf')

        # Extract duals if the model has the 'dual' Suffix and 'LinkingConstraint'
        if hasattr(subproblem_model, 'dual') and subproblem_model.dual.import_enabled():
            if hasattr(subproblem_model, 'LinkingConstraint') and subproblem_model.LinkingConstraint:
                for constraint_index in subproblem_model.LinkingConstraint:
                    try:
                        # Pyomo duals are often non-positive for <= constraints when obj is min.
                        # The Benders cut formula lambda*(x - x_bar) assumes this.
                        dual_val = subproblem_model.dual.get(subproblem_model.LinkingConstraint[constraint_index])
                        if dual_val is not None: # Ensure dual value was actually retrieved
                             linking_constraint_duals[constraint_index] = dual_val
                        # else: dual was not found for this constraint index, do not add to dict
                    except KeyError: # Should not happen if constraint_index is valid
                        pass # linking_constraint_duals[constraint_index] remains unset
            else:
                print("Warning: Subproblem model solved optimally but 'LinkingConstraint' not found for dual extraction.")
        else:
            print("Warning: Subproblem model solved optimally but 'dual' Suffix not found or not enabled.")
    elif term_cond == TerminationCondition.infeasible:
        # Objective is effectively infinite, no useful duals.
        objective_value = float('inf')
        linking_constraint_duals = {}
    # else (error, unbounded, etc.): objective_value remains float('inf'), duals empty.

    return results_object, objective_value, linking_constraint_duals, solver_status, term_cond