# Simulated Annealing
from utils.utils import compute_total_cost
import numpy as np
import time
import random
from heuristics.metaheuristics.neighborhood_operators.two_opt import two_opt_move
from heuristics.metaheuristics.neighborhood_operators.exchange import exchange_move
import math
import copy

def simulated_annealing(instance, routes, min_no_improvement=250, alpha=0.1, beta=0.9): # parameters tuned
    """
    Performs the simulated annealing for VRP

    Return:
        List[List[int]]: best found solution for the VRP problem
    """
    # Initialize
    current_sol = copy.deepcopy(routes)
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    best_sol = current_sol
    best_length = current_length

    temperature = alpha * current_length
    cooling = beta

    max_no_improvement = min_no_improvement

    no_improv = 0
    while no_improv < max_no_improvement:
        improv = False
        # Exploration of the neighborhood
        for it in range(1):
            r_idx_1 = random.randint(0,len(current_sol)-2)
            r_idx_2 = random.randint(0,len(current_sol)-2)
            neighbor, x = exchange_move(current_sol, r_idx_1, r_idx_2, instance)

            neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
            delta = neighbor_length - current_length
            # If the neighbor improves the current solution, we update it
            if delta < 0:
                improv = True
                current_sol = neighbor.copy()
                current_length = neighbor_length
                # If the neighbor is a new best solution, we save it
                if current_length < best_length:
                    best_sol = neighbor.copy()
                    best_length = neighbor_length
            # Update of the current solution even in case of worse neighbors
            elif np.random.random() <  math.exp(-delta / temperature):
                current_sol = neighbor.copy()
                current_length = neighbor_length
        
        if improv:
            no_improv = 0
        else:
            no_improv += 1
        temperature = max(temperature*cooling, 0.0001)

    return best_sol