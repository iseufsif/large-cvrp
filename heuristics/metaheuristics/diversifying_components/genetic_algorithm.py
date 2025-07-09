# Genetic Algorithm
import numpy as np
from utils.utils import compute_total_cost
import copy
import random
from utils.tsp_solvers_for_GA import tsp_solver_ls, tsp_solver_nn

def encoding(routes, instance):
    n = instance["dimension"]
    individual = {"cromosoms":[], "Z": compute_total_cost(routes, instance["edge_weight"]), "f": 0, "p": 0, "range": [0,1],  "feasible": True}
    for node in range(n):
        for route in range(len(routes)):
            if node in routes[route]:
                individual["cromosoms"].append(route)
                break
    return individual

def decoding(child):
    num_vehicles = max(child) + 1
    routes = [[] for _ in range(num_vehicles)]
    for node in range(len(child)):
        for route in range(num_vehicles):
            if child[node] == route:
                routes[route].append(node+1)
    return routes

def parent_selection(population):
    random_num_selection = np.random.random()
    #print(f"Random number: {random_num_selection}")
    for individual in population:
        if random_num_selection > individual["range"][0] and random_num_selection < individual["range"][1]:
            #print(f"Select {individual["cromosoms"]} as first parent")
            break
    parent = copy.copy(individual["cromosoms"])
    return parent

def uniform_crossover(parent_1, parent_2, p = 0.95):
    # Apply uniform crossover:
    child = parent_1
    for i in range(len(parent_1)):
        if np.random.random() < p:
            child[i] = parent_1[i]
        else:
            child[i] = parent_2[i]
    print(f"Crossover performed: {child}")
    return child

def mutation(child, p = 0.5):
    # Apply Mutation
    num_vehicles = max(child) + 1
    for i in range(len(child)):
        if np.random.random() < p:
            child[i] = random.randint(0,num_vehicles-1) # Node assigned to a tour randomly
    print(f"Mutation performed: {child}")
    return child

def capacity_check(routes, instance):
    num_vehicles = len(routes)
    demands = instance["demand"]
    capacity = instance["capacity"]
    loads = [0 for _ in range(num_vehicles)]
    # Compute loads of vehicles
    for route in range(num_vehicles):
        for node in routes[route]:
            loads[route] += demands[node]
            if loads[route] > capacity:
                print("Infeasible Solution")
                return False
    print("Feasible Solution")
    return True

def fitness(population):
    mu = sum(individual["Z"] for individual in population)/len(population)
    for individual in population:
        individual["f"] = 2*mu - individual["Z"]
    
def calculate_probabilities(population):
    # Fitness proportional
    current = 0
    for individual in population:
        individual["p"] = individual["f"] / sum(ind["f"] for ind in population)
        individual["range"] = [current, current + individual["p"]]
        current += individual["p"]


def genetic_algorithm(initial_population, instance, pop_size, max_no_improv = 500):

    # Initialize the population of individuals
    pop = [] 
    for sol in initial_population:
        ind = encoding(sol, instance)
        pop.append(ind)
    fitness(pop)
    calculate_probabilities(pop)
    
    # Parameters
    p_rek = 0.7
    p_mut = 0.3
    gen_size = 20

    no_improv = 0
    it = 1
    best_cost = min(ind["Z"] for ind in pop)

    # Algorithm
    while no_improv < max_no_improv: 
        fitness(pop)
        calculate_probabilities(pop)
        print(f"\nIteration {it}")
        for _ in range(gen_size):
            # Parent Selection
            parent_1 = parent_selection(pop)
            parent_2 = parent_selection(pop)
            # Reprodcution
            if np.random.random() < p_rek:
                child = uniform_crossover(parent_1, parent_2)
            else:
                child = parent_1
                print("No Crossover Performed")
            # Mutation        
            if np.random.random() < p_mut:
                mutation(child)
            else:
                print("No mutation Performed")
            child_decoded = decoding(child)
            # Solve TSP for each route
            new_routes = [] 
            total_length = 0
            for route in child_decoded:
                sequenced_route, route_length = tsp_solver_nn(route, instance["edge_weight"])
                new_routes.append(sequenced_route)  
                total_length += route_length 
            
            # Check the capacity constraint
            feasibility = capacity_check(new_routes, instance)
            if feasibility is True:
                print("Cost New Feasible Solution=", total_length)
            else:
                total_length += 10*total_length # Penalty approach for infeasible solutions
            # Update population
            pop.append({"cromosoms":child, "Z": total_length, "f":0, "p":0, "range":[0,1], "feasible": feasibility})

        # Replacement
        pop = sorted(pop, key=lambda ind: ind["Z"])[:pop_size] #We select the four best solutions
        if round(pop[0]["Z"]) < round(best_cost):
            best_cost = pop[0]["Z"]
            best_sol = pop[0]["cromosoms"]
            no_improv = 0
            print(f"In iteration {it}, populatio = {pop}")
            print(f"In Iteration {it}: Improved to cost {best_cost:.2f}")
        else:
            no_improv += 1
        it +=1

    best_sol_decoded = decoding(best_sol)
    best_sol = []
    best_length = 0
    for route in best_sol_decoded:
        route_sequenced, length = tsp_solver_nn(route, instance["edge_weight"])
        best_sol.append(route_sequenced)
        best_length += length
    return best_sol