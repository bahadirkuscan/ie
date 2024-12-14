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



# This function will handle phase 1 of the 2 phase method
def phase1(Aaux, baux, caux):
    """
    Parameters:
        - Aaux (numpy.ndarray): The auxiliary constraint matrix (m x n).
        Make sure it ends with an identity matrix which will be our initial basis for phase 1
        - baux (numpy.ndarray): The right-hand side vector (m x 1).
        Make sure b wont take any negative value
        - caux (numpy.ndarray): The auxiliary cost vector (1 x n).
        Make sure it is [0,0,0....,0,1,1,...] format. The 0's will correspond to variables of the original problem.

    Returns:
        dict: A dictionary containing the following:
              - 'status': 'optimal', 'infeasible', or 'unbounded'

              - 'z': The value of the auxiliary objective function.
              - 'x': The solution vector for Phase 1.
              IF ARTIFICIAL VARIABLES OF PHASE 1 HAS A VALUE GREATER THAN 0 CONSIDER IT AS UNFEASIBLE
              - 'basis': Indices of the basic variables.
    """



    # Dimensions of the problem
    m, n = Aaux.shape

    # Initialize basis indices (assuming the last m columns correspond to artificial variables)
    basis = list(range(n - m, n))

    # Build the initial tableau
    tableau = np.zeros((m + 1, n + 1))
    tableau[1:, :n] = Aaux # last m row first n col is Aaux
    tableau[1:, -1] = baux.flatten() # last m row last col is baux
    tableau[0, :n] = caux.flatten() # first row first n col is caux
    # first row last col is 0

    # We should substract last m columns from the first one by one to bring it to the standart form
    for i in range(1,m+1):
        tableau[0] -= tableau[i]
    print(tableau)


    while True:
        # Step 1: Check for optimality (all reduced costs >= 0)
        reduced_costs = tableau[0, :-1]
        if np.all(reduced_costs >= 0):
            break

        # Step 2: Choose entering variable (smallest reduced cost)
        entering = np.argmin(reduced_costs) # returns the index of the smallest value

        # Bow we should determine the leaving variable
        ratios = []

        # For each row in constraint matrix
        for i in range(1, m + 1):
            # for valid ratios append them
            if tableau[i, entering] > 0:
                ratios.append(tableau[i, -1] / tableau[i, entering])
            # for invalid ratios append inf to not choose them
            else:
                ratios.append(np.inf)
        # Since we append ratios for each row in constraints the indexes in ratio array corresponds to their places in A,
        # index of leaving variable in tableau is (its index in A) + 1
        leaving = np.argmin(ratios) + 1

        # No leaving variable means I can increase my objective value infinitely, tho I dont know whether I will need it or not in this case
        if ratios[leaving - 1] == np.inf:
            return {
                'status': 'unbounded',
                'z': None,
                'x': None,
                'basis': None,
                'A': None
            }

        # Perform the pivot operation
        pivot = tableau[leaving, entering]
        # Divide leaving variable's row with pivot
        tableau[leaving, :] /= pivot

        # update rest of the tableau
        for i in range(m + 1):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]

        # Update the basis
        basis[leaving - 1] = entering
        print(tableau, "\n\n\n")


    # Write the solution
    x = np.zeros(n)
    for i, b in enumerate(basis):
        x[b] = tableau[i + 1, -1]


    z = tableau[0, -1]
    A = tableau[1:, :n]

    # Check feasibility of the original problem (z must be zero)
    if abs(z) > 1e-6:
        return {
            'status': 'infeasible',
            'z': z,
            'x': x,
            'basis': basis,
            'A': A
        }

    return {
        'status': 'optimal',
        'z': z,
        'x': x,
        'basis': basis,
        'A': A
    }


# Example usage
A = np.array([[0.3, 0.1, 1, 0, 1, 0, 0], [0.5, 0.5, 0, 0, 0, 1, 0], [0.6, 0.4, 0, -1, 0, 0, 1]])
b = np.array([2.7, 6, 6])
c = np.array([0, 0, 0, 0, 1, 1, 1])
mode = -1  # Minimization

result = phase1(A, b, c)
print(result)



