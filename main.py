import TransportationProblem


import numpy as np

# mode = 1 for maximization, -1 for minimization problems
def revised_simplex(c, A, b, mode):
    # Get the row and col count of A
    c = c * mode
    m = len(A)
    n = len(A[0])
    B_indices = [i for i in range(n-m, n)]
    print(B_indices)
    N_indices = [i for i in range(n-m)]

    # Partition A, c into basis and non-basis components
    B = A[:, B_indices]
    N = A[:, N_indices]
    c_B = c[B_indices]
    c_N = c[N_indices]

    # Compute initial basis inverse
    B_inv = np.linalg.inv(B)

    #handle infeasibility
    x_B = B_inv @ b
    if np.any(b < 0):
        return None, "Problem is infeasible"

    while True:
        optimality_test = (c_B @ B_inv @ N) - c_N
        x_B = B_inv @ b
        if np.any(x_B < 0):
            return None, "No feasible solution (basic solution is not feasible)"

        if np.all(optimality_test >= 0):
            # Optimal solution
            x = np.zeros(n)
            x[B_indices] = x_B
            print(B_inv)
            optimal_value = c @ x
            return x, optimal_value * mode


        # find the entering variable
        entering_index = np.argmin(optimality_test)
        entering_var = N_indices[entering_index]

        # find the leaving variable
        d = B_inv @ A[:, entering_var]
        # Check for unboundedness (no leaving variable)
        if np.all(d <= 0):
            return None, "Problem is unbounded"


        # if problem is not unbounded d/x_B will be ratios, min ratio will be leaving variable
        min_value = np.inf
        min_index = -1
        for i in range(m):
            # for invalid d
            if d[i] <= 0:
                continue
            #min ratio test
            ratio = x_B[i] / d[i]
            if ratio < min_value:
                min_value = ratio
                min_index = i


        # the leaving index is the index of leaving variable in B_indices
        leaving_index = min_index
        leaving_var = B_indices[leaving_index]


        # Update the basis
        B_indices[leaving_index] = entering_var
        N_indices[entering_index] = leaving_var

        # Update for the next iteration
        B = A[:, B_indices]
        N = A[:, N_indices]
        c_B = c[B_indices]
        c_N = c[N_indices]
        B_inv = np.linalg.inv(B)




def revised_simplexgpt(A, b, c, mode):
    """
    Solves a linear programming problem in standard form (Ax = b) using the Revised Simplex Method.

    Args:
        A (numpy.ndarray): Coefficient matrix (constraints).
        b (numpy.ndarray): Right-hand side vector (constraints).
        c (numpy.ndarray): Objective function coefficients.
        mode (int): Optimization mode (-1 for minimization, 1 for maximization).

    Returns:
        dict: A dictionary containing the solution and additional information.
    """
    if mode not in [-1, 1]:
        raise ValueError("Mode must be -1 (minimization) or 1 (maximization).")

    # Step 1: Verify feasibility of b
    m, n = A.shape
    if any(b < 0):
        # Phase 1: Solve an auxiliary LP to find an initial feasible solution
        A_aux = np.hstack([A, np.eye(m)])
        c_aux = np.hstack([np.zeros(n), np.ones(m)])
        result_phase1 = revised_simplex_phase1(A_aux, b, c_aux)

        if result_phase1["status"] != "Optimal" or result_phase1["objective_value"] > 1e-8:
            return {"status": "Infeasible"}

        # Extract feasible solution and update basis
        feasible_solution = result_phase1["solution"]
        B = result_phase1["basis"]
        x_B = feasible_solution[:m]
    else:
        # Start with the identity basis if b >= 0
        B = list(range(n, n + m))
        x_B = b.copy()

    # Step 2: Begin the main simplex iterations
    c = mode * c
    N = [i for i in range(n) if i not in B]

    while True:
        # Basis matrix and its inverse
        B_matrix = A[:, B]
        B_inv = np.linalg.inv(B_matrix)

        # Compute reduced costs
        c_B = c[B]
        lambda_ = np.dot(B_inv.T, c_B)
        reduced_costs = c[N] - np.dot(lambda_, A[:, N])

        # Check optimality
        if all(reduced_costs >= 0):
            # Construct full solution
            x = np.zeros(n)
            x[B] = x_B
            return {
                "objective_value": mode * np.dot(c, x),
                "solution": x,
                "status": "Optimal"
            }

        # Select entering variable
        entering_index = np.argmin(reduced_costs)
        entering_variable = N[entering_index]

        # Compute direction vector
        d = np.dot(B_inv, A[:, entering_variable])

        if all(d <= 0):
            return {"status": "Unbounded"}

        # Compute ratios
        ratios = [x_B[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        leaving_index = np.argmin(ratios)
        leaving_variable = B[leaving_index]

        # Update basis
        B[leaving_index] = entering_variable
        N[entering_index] = leaving_variable

        # Update x_B
        x_B = np.dot(B_inv, b)


def revised_simplex_phase1(A, b, c_aux):
    """
    Phase 1 of the Revised Simplex Method to find an initial feasible solution.

    Args:
        A (numpy.ndarray): Auxiliary coefficient matrix.
        b (numpy.ndarray): Right-hand side vector.
        c_aux (numpy.ndarray): Auxiliary objective coefficients.

    Returns:
        dict: Solution and basis for phase 1.
    """
    m, n = A.shape

    # Initial basis with artificial variables
    B = list(range(n - m, n))
    N = list(range(n - m))

    # Compute initial basic feasible solution
    B_matrix = A[:, B]
    x_B = np.linalg.solve(B_matrix, b)

    while True:
        # Basis matrix and reduced costs
        B_matrix = A[:, B]
        B_inv = np.linalg.inv(B_matrix)
        c_B = c_aux[B]
        lambda_ = np.dot(B_inv.T, c_B)
        reduced_costs = c_aux[N] - np.dot(lambda_, A[:, N])

        # Check optimality
        if all(reduced_costs >= 0):
            x = np.zeros(n)
            x[B] = x_B
            return {
                "objective_value": np.dot(c_aux, x),
                "solution": x,
                "basis": B,
                "status": "Optimal"
            }

        # Select entering variable
        entering_index = np.argmin(reduced_costs)
        entering_variable = N[entering_index]

        # Compute direction vector
        d = np.dot(B_inv, A[:, entering_variable])

        if all(d <= 0):
            return {"status": "Unbounded"}

        # Compute ratios
        ratios = [x_B[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))]
        leaving_index = np.argmin(ratios)
        leaving_variable = B[leaving_index]

        # Update basis
        B[leaving_index] = entering_variable
        N[entering_index] = leaving_variable

        # Update x_B
        x_B = np.dot(B_inv, b)


# Example usage
A = np.array([[2, 1, -1, 0], [1, 2, 0, -1]])
b = np.array([2, 3])
c = np.array([1, 2, 0, 0])
mode = -1  # Minimization

result = revised_simplexgpt(A, b, c, mode)
print(result)


