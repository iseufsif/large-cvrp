# implement local search here
import time
import numpy as np
from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.two_opt import two_opt_move

def ls(instance, routes, it=100):
    current_sol = routes.copy()
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    
    # Start LS
    # start_time = time.time()
    improv = True
    while improv:
        improv = False
        
        for r_idx in range(len(current_sol)):
            for _ in range(it):
                neighbor = two_opt_move(current_sol, r_idx)
                neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
                delta = neighbor_length - current_length # calculate delta
                if delta < 0:
                    current_sol = neighbor.copy()       # update x
                    current_length = neighbor_length    # update F(x)
                    improv = True
                    print(f"Route {r_idx} -- Iteration {_}: Improved to cost {current_length:.2f}")
    return current_sol



