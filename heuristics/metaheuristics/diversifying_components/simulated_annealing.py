# Simulated Annealing
from utils.utils import compute_total_cost
import numpy as np
import time
from heuristics.metaheuristics.neighborhood_operators.two_opt import two_opt_move
import math

def simulated_annealing(instance, routes, it=50, max_no_improvement=200, alpha=0.1, beta=0.9):
    # Initialize
    current_sol = routes.copy()
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    best_sol = current_sol
    best_length = current_length

    temperature = alpha * current_length
    cooling = beta

    # Set seed to reproduce random based results
    # np.random.seed(1)

    # Start LS
    # start_time = time.time()
    no_improv = 0
    while no_improv < max_no_improvement:
        improv = False
        for r_idx in range(len(current_sol)):
            for _ in range(it):
                neighbor = two_opt_move(current_sol, r_idx)
                neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
                delta = neighbor_length - current_length
                if delta < 0:
                    improv = True
                    current_sol = neighbor.copy()
                    current_length = neighbor_length
                    if current_length < best_length:
                        best_sol = neighbor.copy()
                        best_length = neighbor_length
                        print(f"Route {r_idx}: New best solution value: {best_length}")
                elif np.random.random() <  math.exp(-delta / temperature):
                    current_sol = neighbor.copy()
                    current_length = neighbor_length
        
        if improv:
            no_improv = 0
        else:
            no_improv += 1
        temperature = temperature * cooling

    return best_sol