import random
import numpy as np

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value


class TransportationProblem:
    def __init__(self, supply, demand, cost_matrix):
        self.supply = supply
        self.demand = demand
        self.cost_matrix = cost_matrix

    @staticmethod
    def generate_instance(supply_nodes, demand_nodes, max_cost, max_supply_demand):
        np.random.seed(0)
        supply = np.random.randint(1, max_supply_demand + 1, size=supply_nodes)
        demand = np.random.randint(1, max_supply_demand + 1, size=demand_nodes)

        # Total supply = total demand
        total_supply = sum(supply)
        total_demand = sum(demand)
        difference = total_supply - total_demand
        while difference > 0:
            index = random.randint(0, supply_nodes - 1)
            reduced_amount = random.randint(0, min(difference, supply[index]) + 1)
            difference -= reduced_amount
            supply[index] -= reduced_amount
        while difference < 0:
            index = random.randint(0, demand_nodes - 1)
            increased_amount = random.randint(0, min(difference, demand[index]) + 1)
            difference += increased_amount
            demand[index] -= increased_amount


        cost_matrix = np.random.randint(1, max_cost + 1, size=(supply_nodes, demand_nodes))
        return TransportationProblem(supply, demand, cost_matrix)

    def solve_with_solver(self):
        """
        Solves the transportation problem using an LP solver (e.g., Pulp).

        :return: Optimal value and the solution as a dictionary.
        """
        supply_nodes = range(len(self.supply))
        demand_nodes = range(len(self.demand))

        # Create LP problem
        prob = LpProblem("TransportationProblem", LpMinimize)

        # Decision variables
        x = LpVariable.dicts("x", ((i, j) for i in supply_nodes for j in demand_nodes), lowBound=0, cat="Continuous")

        # Objective function
        prob += lpSum(self.cost_matrix[i][j] * x[i, j] for i in supply_nodes for j in demand_nodes)

        # Supply constraints
        for i in supply_nodes:
            prob += lpSum(x[i, j] for j in demand_nodes) == self.supply[i]

        # Demand constraints
        for j in demand_nodes:
            prob += lpSum(x[i, j] for i in supply_nodes) == self.demand[j]

        # Solve
        prob.solve()

        # Extract solution
        solution = {k: v.varValue for k, v in x.items()}
        return value(prob.objective), solution

    def __repr__(self):
        return (f"Supply: {self.supply}\n"
                f"Demand: {self.demand}\n"
                f"Cost Matrix:\n{np.array(self.cost_matrix)}")
