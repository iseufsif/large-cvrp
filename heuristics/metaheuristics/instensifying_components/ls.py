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
        for i in range(it):
            index_1 = np.random.randint(1, len(current_sol)-2) # select two indices for two-Opt
            index_2 = np.random.randint(index_1, len(current_sol)-1)
            neighbor = two_opt_move(current_sol, index_1, index_2)
            neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
            delta = neighbor_length - current_length # calculate delta
            if delta < 0:
                current_sol = neighbor.copy()       # update x
                current_length = neighbor_length    # update F(x)
                improv = True
                print(f"Iteration: {i} -- Current Best Solution: {current_length}")
    return current_sol



