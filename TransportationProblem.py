import random
import numpy as np



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
            reduced_amount = random.randint(0, min(difference, supply[index]))
            difference -= reduced_amount
            supply[index] -= reduced_amount
        while difference < 0:
            index = random.randint(0, demand_nodes - 1)
            increased_amount = random.randint(0, min(-difference, demand[index]))
            difference += increased_amount
            demand[index] -= increased_amount


        cost_matrix = np.random.randint(1, max_cost + 1, size=(supply_nodes, demand_nodes))
        return TransportationProblem(supply, demand, cost_matrix)



    def __repr__(self):
        return (f"Supply: {self.supply}\n"
                f"Demand: {self.demand}\n"
                f"Cost Matrix:\n{np.array(self.cost_matrix)}")
