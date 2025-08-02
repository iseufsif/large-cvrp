import copy
import numpy as np
from utils.utils import compute_route_cost
import random

def two_opt_move(routes, route_idx, edge_weight):
    """
    Performs intra-route two-opt to a route of a VRP solution

    Return:
        List[List[int]]: modified VRP solution
        Float: delta cost from the previous to the new solution
    """
    if not routes or len(routes[route_idx]) < 4:
        return routes, 0  # too short to optimize

    route = routes[route_idx]
    n = len(route)

    # Random i, j with j > i + 1 to ensure nontrivial 2-opt
    i = np.random.randint(0, n - 2)
    j = np.random.randint(i + 2, n)

    new_route = route[:i] + list(reversed(route[i:j])) + route[j:]
    
    # Compute cost delta for this route only
    old_cost = compute_route_cost(route, edge_weight)
    new_cost = compute_route_cost(new_route, edge_weight)
    delta = new_cost - old_cost

    new_routes = copy.deepcopy(routes)
    new_routes[route_idx] = new_route

    return new_routes, delta