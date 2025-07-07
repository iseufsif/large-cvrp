# Genetic Algorithm
import numpy as np
from utils.utils import compute_total_cost

def encoding(routes, instance):
    n = instance["dimension"]
    individual = {"cromosoms":[], "Z": compute_total_cost(routes, instance["edge_weight"]), "f": 0, "p": 0, "range": [0,1],  "feasbile": True}
    for node in range(n):
        for route in range(len(routes)):
            if node in routes[route]:
                individual["cromosoms"].append(route)
                break
    return individual

def decoding(child, num_vehicles):
    routes = [[] for _ in range(num_vehicles)]
    for node in range(len(child)):
        for route in range(num_vehicles):
            if child[node] == route:
                routes[route].append(node+1)
    return routes

def uniform_crossover(parent_1, parent_2, p = 0.5):
    # Apply uniform crossover:
    child = parent_1
    for i in range(len(parent_1)):
        if np.random.random() < p:
            child[i] = parent_1[i]
        else:
            child[i] = parent_2[i]
    print(f"Crossover performed: {child}")
    return child

def mutation(child, p = 0.3):
    # Apply Mutation
    for i in range(len(child)):
        if np.random.random() < p:
            child[i] = 1 - child[i]
    print(f"Mutation performed: {child}")
    return child

def capacity_check(individual, instance, num_vehicles):
    demands = instance["demand"]
    capacity = instance["capacity"]
    loads = [0 for _ in len(num_vehicles)]

    # Compute loads of vehicles
    for node in range(len(individual["cromosoms"])):
        for route in range(num_vehicles):
            if individual["cromosoms"][node] == route:
                loads[route] += demands[node+1]
    # Check capacity
    for route in range(len(loads)):
        if loads[route] > capacity:
            individual["feasible"] = False
            break
    return None

def fitness(population):
    mu = sum(individual["Z"] for individual in population)/len(population)
    for individual in population:
        individual["f"] = 2*mu - individual["Z"]
        if individual["feasible"] == False:
            individual["f"] -= mu # Penalty approach to treat infeasibility, we can change that

def calculate_probabilities(population):
    # Fitness proportional
    current = 0
    for individual in population:
        individual["p"] = individual["f"] / sum(ind["f"] for ind in population)
        individual["range"] = [current, current + individual["p"]]
        current += individual["p"]

def tsp_solver(child):
    # We must define a way to solve the tsp for each route after having genereted a new child
    # Otherwise the sequence of visiting nodes will just follow an increasing order of the nodes' id
    return None