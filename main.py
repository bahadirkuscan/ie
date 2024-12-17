import random

import Transportation_Problem

import cvxpy as cp

import numpy as np


def transportation_to_lp(supply, demand, cost_matrix):
    """
    Converts a transportation problem into LP form: min c^T x st Ax = b.

    Args:
        supply (list or numpy.ndarray): Supply at each supply node.
        demand (list or numpy.ndarray): Demand at each demand node.
        cost_matrix (list or numpy.ndarray): Cost matrix (m x n) where m = supply nodes, n = demand nodes.

    Returns:
        dict: A dictionary containing:
            - c (numpy.ndarray): Objective function coefficients.
            - A (numpy.ndarray): Constraint matrix.
            - b (numpy.ndarray): Right-hand side vector.
    """
    # Number of supply and demand nodes
    m, n = len(supply), len(demand)

    # Flatten the cost matrix to create c
    c = cost_matrix.flatten()

    # Create the constraint matrix A and vector b
    A = []
    b = []

    # Supply constraints
    for i in range(m):
        row = np.zeros(m * n)
        for j in range(n):
            row[i * n + j] = 1
        A.append(row)
        b.append(supply[i])

    # Demand constraints
    for j in range(n):
        row = np.zeros(m * n)
        for i in range(m):
            row[i * n + j] = 1
        A.append(row)
        b.append(demand[j])

    A = np.array(A)
    b = np.array(b)

    return {
        "c": c,
        "A": A,
        "b": b
    }


# mode = 1 for maximization, -1 for minimization problems
def revised_simplex(A, b, c):
    """
    Solves a linear programming problem in standard form (Ax = b) using the Revised Simplex Method.

    Args:
        A (numpy.ndarray): Coefficient matrix (constraints).
        b (numpy.ndarray): Right-hand side vector (constraints).
        c (numpy.ndarray): Objective function coefficients.

    Returns:
        dict: A dictionary containing the solution and additional information.
    """
    # Step 1: Verify feasibility of b
    m, n = A.shape
    x_B = list(range(n - m, n))
    # x_N = indexes of nonbasic variables in c
    # x_B = indexes of basic variables in c and any row of A
    # b = right hand side
    x_N = list(range(n - m))

    while True:
        # B = the columns corresponds to basic variables in A
        # N = matrix corresponds to nonbasic variables in A
        B = A[:, x_B]
        N = A[:, x_N]
        # Basis matrix and its inverse
        B_inv = np.linalg.inv(B)

        # Make an optimality test
        c_B = c[x_B]
        optimality_values = c_B @ B_inv @ N - c[x_N]
        # Check optimality
        if all(optimality_values >= 0):
            opt_variables = np.zeros(n)
            opt_variables[x_B] = B_inv @ b.T
            return {
                "x_B": x_B,
                "objective_variables": opt_variables,
                "objective_value": c_B @ B_inv @ b.T, # since b is represented as a 1D array for convention take flip it
                "status": "Optimal"
            }

        # Select entering variable
        entering_index = np.argmin(optimality_values) # entering_index is index of entering variable in x_N
        entering_variable = x_N[entering_index] # entering_variable is its index in c

        # Compute direction vector
        d = np.dot(B_inv, A[:, entering_variable])

        if all(d <= 0.00000001):
            return {"status": "Unbounded"}

        # Find the ratios
        ratios = np.zeros(m)
        for i in range(m):
            if d[i] > 0.000001:
                ratios[i] = (np.dot(B_inv, b))[i] / d[i]
            else: ratios[i] = np.inf

        # min ratio test
        leaving_index = np.argmin(ratios) # leaving_index is index of leaving variable in x_B
        leaving_variable = x_B[leaving_index] # leaving_variable is its index in c

        # Update basis
        x_B[leaving_index] = entering_variable
        x_N[entering_index] = leaving_variable




def solve(A,b,c,mode):
    """
    finds optimal z = cx subject to Ax = b by utilizing big M method and revised simplex
    :param mode: 1 for maximization, -1 for minimization
    """
    if mode != 1 and mode != -1:
        print("invalid input mode")
        return

    m,n = A.shape
    # Since negative values on b will result infeasibility at revised simplex stage we will multiply entire row with -1
    if any(b < 0):
        for i in range(m):
            if b[i] < 0:
                A[i, :] *= -1
                b[i] *= -1
    # Since revised simplex is written for maximization problem convert minimization into maximization
    c *= mode
    # Add an identity matrix at the end of A for big M method
    A_aux = np.hstack([A, np.eye(m)])
    # Find the M which is 1000 times the max value in problem
    M = 10**6
    c = np.append(c, [-M] * m)
    # Bring it into canonical form
    for i in range(m):
        for j in range(m+n):
            c[j] += A_aux[i][j] * M
    result = revised_simplex(A_aux,b,c)
    # Compute feasibility
    if result["status"] == "Optimal":
        objective = result["objective_variables"]
        x_B = result["x_B"]
        # Artificial variables should be zero
        for i in range(m):
            if x_B[i] >= n:
                if abs(objective[x_B[i]]) > 0.000001:
                    return {
                        "infeasible"
                    }

        return {
            "status": "Optimal",
            "objective_variables": objective[:m+n],
            "objective_value": (result["objective_value"] - M * np.sum(b)) * mode
        }

    else:
        return result









max_cost = 50
for i in range(10):
    node_count = random.randint(2, 10)
    instance = Transportation_Problem.TransportationProblem.generate_instance(node_count, node_count, max_cost, max_cost)
    lp = transportation_to_lp(instance.supply, instance.demand, instance.cost_matrix)

    x = cp.Variable(node_count ** 2)
    constraints = [lp["A"] @ x == lp["b"], x >= 0]
    objective = cp.Minimize(lp["c"] @ x)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()


    print("Cvxpy solver result:", result)


    print("Our solver result:" ,solve(lp["A"], lp["b"], lp["c"], -1)["objective_value"])
    print("\n\n")



