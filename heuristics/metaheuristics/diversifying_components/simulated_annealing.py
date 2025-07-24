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
    # Initialize
    current_sol = copy.deepcopy(routes)
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    best_sol = current_sol
    best_length = current_length

    temperature = alpha * current_length
    cooling = beta

    max_no_improvement = min_no_improvement
    # Start LS
    # start_time = time.time()
    no_improv = 0
    while no_improv < max_no_improvement:
        improv = False
        for it in range(1):
            """if len(current_sol[r_idx]) < 3:
                continue  # skip short routes"""
            #n = len(current_sol[r_idx])
            #for _ in range(round(1.5*n)): # Tuned
                #neighbor, x = two_opt_move(current_sol, r_idx, instance["edge_weight"])
            r_idx_1 = random.randint(0,len(current_sol)-2)
            r_idx_2 = random.randint(0,len(current_sol)-2)
            neighbor, x = exchange_move(current_sol, r_idx_1, r_idx_2, instance)

            neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
            delta = neighbor_length - current_length
            if delta < 0:
                improv = True
                current_sol = neighbor.copy()
                current_length = neighbor_length
                if current_length < best_length:
                    best_sol = neighbor.copy()
                    best_length = neighbor_length
                    # print(f"Route {r_idx}: New best solution value: {best_length}")
            elif np.random.random() <  math.exp(-delta / temperature):
                current_sol = neighbor.copy()
                current_length = neighbor_length
        
        if improv:
            no_improv = 0
        else:
            no_improv += 1
        temperature = max(temperature*cooling, 0.0001)

    return best_sol