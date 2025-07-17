import random
from utils.utils import compute_route_cost

def exchange_move(routes, route_1_idx, route_2_idx, instance):    
    capacity = instance["capacity"]
    demands = instance["demand"]
    edge_weight = instance["edge_weight"]

    route1 = routes[route_1_idx]
    route2 = routes[route_2_idx]

    if not route1 or not route2:
        return routes, 0

    index_1 = random.randint(0, len(route1) - 1)
    index_2 = random.randint(0, len(route2) - 1)

    node1 = route1[index_1]
    node2 = route2[index_2]

    load1 = sum(demands[i] for i in route1) - demands[node1] + demands[node2]
    load2 = sum(demands[i] for i in route2) - demands[node2] + demands[node1]

    if load1 > capacity or load2 > capacity:
        return routes, 0  # infeasible

    # Apply swap
    new_route1 = route1.copy()
    new_route2 = route2.copy()
    new_route1[index_1] = node2
    new_route2[index_2] = node1

    # Compute delta cost
    old_cost_1 = compute_route_cost(route1, edge_weight)
    old_cost_2 = compute_route_cost(route2, edge_weight)
    new_cost_1 = compute_route_cost(new_route1, edge_weight)
    new_cost_2 = compute_route_cost(new_route2, edge_weight)

    delta = (new_cost_1 + new_cost_2) - (old_cost_1 + old_cost_2)

    # Build new routes list
    new_routes = routes.copy()
    new_routes[route_1_idx] = new_route1
    new_routes[route_2_idx] = new_route2

    return new_routes, delta