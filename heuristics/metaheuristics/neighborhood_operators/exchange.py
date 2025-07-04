import copy
import numpy as np
import random

def exchange_move(routes, route_1_idx, route_2_idx, instance):
    
    capacity = instance["capacity"]
    demands = instance["demand"]
    
    new_routes = copy.deepcopy(routes)
    
    if route_1_idx == route_2_idx:
        return routes

    index_1 = random.randint(0, len(new_routes[route_1_idx])-1)
    index_2 = random.randint(0, len(new_routes[route_2_idx])-1)

    node1 = new_routes[route_1_idx][index_1]
    node2 = new_routes[route_2_idx][index_2]

    load1 = sum(demands[i] for i in new_routes[route_1_idx])
    load2 = sum(demands[i] for i in new_routes[route_2_idx])
    
    if load1 - demands[node1] + demands[node2] <= capacity and load2 - demands[node2] + demands[node1] <= capacity:
        new_routes[route_1_idx][index_1] = node2
        new_routes[route_2_idx][index_2] = node1
    
    return new_routes