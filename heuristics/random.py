import vrplib
import numpy as np
import random
import os

def generate_random_solution(instance_name):
    # load instance
    instance = vrplib.read_instance("instances/" + instance_name)

    # load parameters
    capacity = instance["capacity"]
    demands = instance["demand"]  # demand[0] is depot
    customers = list(range(1, len(demands)))  # customer indices start from 1

    random.shuffle(customers)

    routes = []
    current_route = []
    current_load = 0

    for customer in customers:
        if current_load + demands[customer] <= capacity:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            routes.append(current_route)
            current_route = [customer]
            current_load = demands[customer]
    
    if current_route:
        routes.append(current_route)

    return routes, instance
