import random
import copy
from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.two_opt import two_opt_move
from heuristics.metaheuristics.neighborhood_operators.exchange import exchange_move

def ls_with_2opt(instance, routes, it=100):
    """
    Performs local search solve the VRP problem
    Neighborhood operator: Two-Opt

    Return:
        List[List[int]]: best found solution to VRP
    """
    current_sol = copy.deepcopy(routes)
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    
    improv = True
    while improv:
        improv = False
        
        for r_idx in range(len(current_sol)):
            if len(current_sol[r_idx]) < 3:
                continue  # skip too-short routes
            for _ in range(it):
                neighbor, delta = two_opt_move(current_sol, r_idx, instance["edge_weight"])
                neighbor_length = current_length + delta
                if delta < 0:
                    current_sol = copy.deepcopy(neighbor)
                    current_length = neighbor_length
                    improv = True
    return current_sol

def ls_with_swaps(instance, routes, it=100):
    """
    Performs local search solve the VRP problem
    Neighborhood operator: Exchange

    Return:
        List[List[int]]: best found solution to VRP
    """
    current_sol = copy.deepcopy(routes)
    current_length = compute_total_cost(current_sol, instance["edge_weight"])
    
    improv = True
    iter = 1
    while improv:
        improv = False
        for _ in range(it):
            route_1_idx = random.randint(0,len(current_sol)-1)
            route_2_idx = random.randint(0,len(current_sol)-1)
            neighbor, delta = exchange_move(current_sol, route_1_idx, route_2_idx, instance)
            neighbor_length = current_length + delta
            if delta < 0:
                current_sol = copy.deepcopy(neighbor)
                current_length = neighbor_length
                improv = True
        iter += 1
    return current_sol

def hybrid_ls(instance, routes, it=100):
    """
    Performs local search solve the VRP problem
    Neighborhood operator: combined use of Exchange and Two-Opt

    Return:
        List[List[int]]: best found solution to VRP
    """
    ls_inter_route = ls_with_swaps(instance, routes, it)
    ls_intra_route = ls_with_2opt(instance, ls_inter_route, it)
    return ls_intra_route
