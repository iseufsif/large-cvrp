# Genetic Algorithm
import numpy as np
from utils.utils import compute_total_cost
import copy
import random
from utils.tsp_solvers_for_GA import tsp_solver_ls, tsp_solver_nn
from heuristics.construction.random import generate_random_solution
from heuristics.metaheuristics.instensifying_components.ls import hybrid_ls

def split(permutation, demand, capacity):
    routes = []
    route = []
    load = 0
    for node in permutation:
        if load + demand[node] <= capacity:
            route.append(node)
            load += demand[node]
        else:
            routes.append(route)
            route = [node]
            load = demand[node]
    if route:
        routes.append(route)
    return routes
    

def parent_selection(population):
    # Rank based selecton
    pop_sorted = sorted(population, key=lambda x: x["fitness_combined"])

    for rank, individual in enumerate(pop_sorted):
        individual["total_rank"] = rank
    
    random_num_selection = np.random.random()
    for individual in population:
        if random_num_selection > individual["range"][0] and random_num_selection < individual["range"][1]:
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
    return child

def order_crossover(parent1, parent2):
    # Apply order crossover
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    current_index = (end + 1) % size
    parent2_index = (end + 1) % size

    while None in child:
        gene = parent2[parent2_index]
        if gene not in child:
            child[current_index] = gene
            current_index = (current_index + 1) % size
        parent2_index = (parent2_index + 1) % size

    return child


def mutation(child, p = 0.2):
    # Apply Mutation by swapping a node with another random node 
    n = len(child)
    for i in range(n):
        if np.random.random() < p:
            node_1 = child[i]
            rand_index = random.randint(0,n-1)
            node_2 = child[rand_index]
            child[i] = node_2
            child[rand_index] = node_1

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
    E_max = 1.5 #tuned
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

def genetic_algorithm(instance, pop_size, max_no_improv = 100):

    # Initialize the population of individuals
    pop = []
    for i in range(pop_size):
        individual = {
        "cromosoms": np.random.permutation(list(range(1, instance["dimension"]))).tolist(),
        "Z": 0,
        "f": 0,
        "div": 0,
        "p": 0,
        "range": [0, 1],
        "feasible": True}
        sol = split(individual["cromosoms"], instance["demand"], instance["capacity"])
        #sol = hybrid_ls(instance, sol, 50) # Educate initial random population with hybrid ls
        #individual["cromosoms"] = [node for route in sol for node in route]
        individual["Z"] = compute_total_cost(sol, instance["edge_weight"])
        individual["feasible"] = capacity_check(sol, instance)
        pop.append(individual)
    
    n_elite = 4
    fitness_quality(pop)
    diversity(pop)
    calculate_combined_fitness(pop, n_elite)
    calculate_probabilities(pop)

    # Parameters
    p_rek = 0.7 # tuned
    p_mut = 0.2 # tuned
    gen_size = 25 # from literature
    penalty = 10

    no_improv = 0
    it = 1
    n = instance["dimension"]
    best_cost = min(ind["Z"] for ind in pop)

    # Algorithm
    while no_improv < max_no_improv: 
        new_best_sol_found = False
        fitness_quality(pop)
        if it%round(0.1*n) == 0: # tuned
            diversity(pop)
        calculate_combined_fitness(pop, n_elite)
        calculate_probabilities(pop)
        #print(f"\nIteration {it}")
        for _ in range(gen_size):
            # Parent Selection
            parent_1 = parent_selection(pop)
            parent_2 = parent_selection(pop)
            # Reprodcution
            if np.random.random() < p_rek:
                child = order_crossover(parent_1, parent_2)
            else:
                child = parent_1
            # Mutation        
            if np.random.random() < p_mut:
                mutation(child)

            # Check if the child is a clone ()
            seen = set(tuple(ind["cromosoms"]) for ind in pop)
            if tuple(child) in seen:
                continue

            routes = split(child, instance["demand"], instance["capacity"])

            # Educate the child with nearest neighbor
            new_routes = [] 
            total_length = 0
            for route in routes:
                sequenced_route, route_length = tsp_solver_nn(route, instance["edge_weight"])
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
                total_length = penalty*total_length
            
            # Update population
            pop.append({"cromosoms": [node for route in new_routes for node in route], # Encoding of the child educated with nearest neighbor
                        "Z": total_length,
                        "f":0, 
                        "div":0, 
                        "fitness_combined":0, 
                        "p":0, 
                        "range":[0,1], 
                        "feasible": feasibility})

        # Replacement
        pop = sorted(pop, key=lambda ind: ind["Z"])[:pop_size+gen_size]
        
        # Improvement Management
        if new_best_sol_found is True:
            no_improv = 0
        else:
            no_improv += 1

        # After 250 iteration with no improvement, we replace some individuals with random solutions
        if no_improv > (2.5*n) and no_improv%round(0.25*n) == 0: # Tuned
            num_replace = int(pop_size * 0.2)
            new_individuals = []
            for _ in range(num_replace):
                ind = {
                    "cromosoms": np.random.permutation(list(range(1, instance["dimension"]))).tolist(),
                    "Z": 0,
                    "f": 0,
                    "div": 0,
                    "p": 0,
                    "range": [0, 1],
                    "feasible": True}
                sol = split(ind["cromosoms"], instance["demand"], instance["capacity"])
                ind["Z"] = compute_total_cost(sol, instance["edge_weight"])
                ind["feasible"] = capacity_check(sol, instance)
                new_individuals.append(ind)
            pop = pop[:-num_replace] + new_individuals
        print(no_improv)
        it +=1

    return best_sol