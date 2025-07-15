# Genetic Algorithm
import numpy as np
from utils.utils import compute_total_cost
import copy
import random
from utils.tsp_solvers_for_GA import tsp_solver_ls, tsp_solver_nn
from heuristics.construction.random import generate_random_solution

def encoding(routes, instance):
    n = instance["dimension"]
    individual = {"cromosoms":[], "Z": compute_total_cost(routes, instance["edge_weight"]), "f": 0, "div": 0, "p": 0, "range": [0,1],  "feasible": True}
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

    pop_sorted = sorted(population, key=lambda x: x["fitness_combined"])

    for rank, individual in enumerate(pop_sorted):
        individual["total_rank"] = rank
    
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
    child = copy.deepcopy(parent_1)
    for i in range(len(parent_1)):
        if np.random.random() < p:
            child[i] = parent_1[i]
        else:
            child[i] = parent_2[i]
    #print(f"Crossover performed: {child}")
    return child

def mutation(child, p = 0.5):
    # Apply Mutation
    num_vehicles = max(child) + 1
    for i in range(len(child)):
        if np.random.random() < p:
            child[i] = random.randint(0,num_vehicles-1) # Node assigned to a tour randomly
    #print(f"Mutation performed: {child}")
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
                return False
    #print("Feasible Solution")
    return True

def fitness_quality(population):
    mu = sum(individual["Z"] for individual in population)/len(population)
    for individual in population:
        individual["f"] = 2*mu - individual["Z"]

def calculate_combined_fitness(population, n_elite):
    pop_size = len(population)
    
    # Rank for quality (lower Z is better)
    sorted_by_quality = sorted(population, key=lambda x: x["Z"])
    for rank, individual in enumerate(sorted_by_quality):
        individual["rank_quality"] = rank
    
    # Rank for diversity (higher div is better → invert sort)
    sorted_by_div = sorted(population, key=lambda x: -x["div"])
    for rank, individual in enumerate(sorted_by_div):
        individual["rank_div"] = rank

    # Compute combined fitness score
    for individual in population:
        f_quality = individual["rank_quality"]
        f_div = individual["rank_div"]
        individual["fitness_combined"] = f_div + (1 - n_elite / pop_size) * f_quality
    
def calculate_probabilities(population):
    # Rank based selection
    E_max = 1.2
    E_min = 2 - E_max
    pop_size = len(population)
    pop_sorted = sorted(population, key=lambda x: x["fitness_combined"])
    for rank, individual in enumerate(pop_sorted):
        individual["total_rank"] = rank        
    current = 0
    for individual in population:
        individual["p"] = (E_max - (E_max - E_min)*((individual["total_rank"]-1)/(pop_size-1)))/pop_size
        individual["range"] = [current, current + individual["p"]]
        current += individual["p"]

def is_duplicate(child, population):
    return any(child == ind["cromosoms"] for ind in population)

def diversity(population):
    for individual in population:
        div = 0
        for other in population:
            if individual == other:
                continue
            diff = sum([1 for a, b in zip(individual["cromosoms"], other["cromosoms"]) if a != b])
            div += diff / len(individual["cromosoms"])
        individual["div"] = div / (len(population) - 1)

def genetic_algorithm(initial_population, instance, pop_size, instance_name, max_no_improv = 2000):

    # Initialize the population of individuals
    pop = [] 
    n_elite = 1
    for sol in initial_population:
        ind = encoding(sol, instance)
        pop.append(ind)
    fitness_quality(pop)
    diversity(pop)
    calculate_combined_fitness(pop, n_elite)
    calculate_probabilities(pop)
    print(pop)
    # Parameters
    p_rek = 0.7
    p_mut = 0.3
    gen_size = 25

    no_improv = 0
    it = 1
    best_cost = min(ind["Z"] for ind in pop)

    # Algorithm
    while no_improv < max_no_improv: 
        new_best_sol_found = False
        fitness_quality(pop)
        diversity(pop)
        calculate_combined_fitness(pop, n_elite)
        calculate_probabilities(pop)
        #print(f"\nIteration {it}")
        for _ in range(gen_size):
            # Parent Selection
            parent_1 = parent_selection(pop)
            parent_2 = parent_selection(pop)
            # Reprodcution
            if np.random.random() < 1:
                child = uniform_crossover(parent_1, parent_2)
            else:
                child = parent_1
            # Mutation        
            if np.random.random() < p_mut:
                mutation(child)
            if is_duplicate(child, pop):
                continue
            child_decoded = decoding(child)
            # Solve TSP for each route
            new_routes = [] 
            total_length = 0
            for route in child_decoded:
                sequenced_route, route_length = tsp_solver_ls(route, instance["edge_weight"])
                new_routes.append(sequenced_route)  
                total_length += route_length 
            
            # Check the capacity constraint
            feasibility = capacity_check(new_routes, instance)
            if feasibility is True:
                #print("Cost New Feasible Solution=", total_length)
                if total_length < best_cost:
                    best_cost = total_length
                    best_sol = new_routes
                    new_best_sol_found = True
                    #print(f"In Iteration {it}: Improved to cost {best_cost:.2f}")
            else:
                total_length = 10*total_length # Penalty approach for infeasible solutions
            # Update population
            
            pop.append({"cromosoms":child, "Z": total_length, "f":0, "div": 0, "fitness_combined" : 0, "p":0, "range":[0,1], "feasible": feasibility})

        # Replacement
        pop = sorted(pop, key=lambda ind: ind["Z"])[:pop_size+gen_size]
        #print(f"In iteration {it}, population = {pop}")
        
        # Improvement Management
        if new_best_sol_found is True:
            no_improv = 0
        else:
            no_improv += 1

        if no_improv > 250:
            num_replace = int(pop_size * 0.2)
            new_individuals = []
            for _ in range(num_replace):
                sol, inst = generate_random_solution(instance_name)
                ind = encoding(sol, instance)
                new_individuals.append(ind)
            pop = pop[:-num_replace] + new_individuals
        print(no_improv)
        #print(pop)
        it +=1
    
    return best_sol