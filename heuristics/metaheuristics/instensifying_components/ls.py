# implement local search here
import time
import numpy as np
import random
import copy
from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.two_opt import two_opt_move
from heuristics.metaheuristics.neighborhood_operators.exchange import exchange_move

def ls_with_2opt(instance, routes, it=100):
    current_sol = copy.deepcopy(routes)
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
                    # print(f"Route {r_idx} -- Iteration {_}: Improved to cost {current_length:.2f}")
    return current_sol

def ls_with_swaps(instance, routes, it=100):
    current_sol = copy.deepcopy(routes)
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    
    # Start LS
    # start_time = time.time()
    improv = True
    iter = 1
    while improv:
        improv = False
        for _ in range(it):
            route_1_idx = random.randint(0,len(current_sol)-1)
            route_2_idx = random.randint(0,len(current_sol)-1)
            neighbor = exchange_move(current_sol, route_1_idx, route_2_idx, instance)
            neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])
            delta = neighbor_length - current_length # calculate delta
            if delta < 0:
                current_sol = neighbor.copy()       # update x
                current_length = neighbor_length    # update F(x)
                improv = True
                # print(f"Swap among routes {route_1_idx} and {route_2_idx} in Iteration {iter}.{_}: Improved to cost {current_length:.2f}")
        iter += 1
    return current_sol

def hybrid_ls(instance, routes, it=100):
    ls1 = ls_with_swaps(instance, routes, it)
    ls2 = ls_with_2opt(instance, ls1, it)
    return ls2
