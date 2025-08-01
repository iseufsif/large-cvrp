from heuristics.metaheuristics.diversifying_components.genetic_algorithm import split,fitness_quality, calculate_probabilities, parent_selection, order_crossover,capacity_check, calculate_combined_fitness, diversity
from utils.tsp_solvers_for_GA import tsp_solver_nn
from heuristics.improvement.ls import hybrid_ls
import numpy as np
from utils.utils import compute_total_cost

def HGS(instance, pop_size, max_no_improv = 100):
    # Initialize the population of individuals
    pop = [] 
    for i in range(pop_size):
        individual = {
        "cromosoms": np.random.permutation(list(range(1, instance["dimension"]))).tolist(),
        "Z": 0,
        "f": 0,
        "div": 0,
        "fitness_combined": 0,
        "p": 0,
        "range": [0, 1],
        "feasible": True}
        sol = split(individual["cromosoms"], instance["demand"], instance["capacity"])
        individual["Z"] = compute_total_cost(sol, instance["edge_weight"])
        individual["feasible"] = capacity_check(sol, instance)
        pop.append(individual)

    n_elite = 4

    fitness_quality(pop)
    diversity(pop)
    calculate_combined_fitness(pop, n_elite)
    calculate_probabilities(pop)
    
    gen_size = 25
    penalty = 10
    no_improv = 0
    it = 1
    it_ls = 0
    best_cost = min(ind["Z"] for ind in pop)

    # Algorithm
    while no_improv < max_no_improv: 
        new_best_sol_found = False
        fitness_quality(pop)
        if it%5 == 0:
            diversity(pop)
        calculate_combined_fitness(pop, n_elite)
        calculate_probabilities(pop)
        #print(f"\nIteration {it}")
        for _ in range(gen_size):

            # Parent Selection
            parent_1 = parent_selection(pop)
            parent_2 = parent_selection(pop)
    
            # Reprodcution
            child = order_crossover(parent_1, parent_2)

            #Check if the child is a clone
            seen = set(tuple(ind["cromosoms"]) for ind in pop)
            if tuple(child) in seen:
                continue

            routes = split(child, instance["demand"], instance["capacity"])

            # Educate child using Nearest Neighbor and Local Search
            new_routes = [] 
            total_length = 0
            for route in routes:
                sequenced_route, route_length = tsp_solver_nn(route, instance["edge_weight"])
                new_routes.append(sequenced_route)
                total_length += route_length  

            if  total_length > 1.5 *best_cost:
                continue  # skip LS on bad offsprings
            else:
                child_ls = hybrid_ls(instance, new_routes, min(10, 3 + it_ls))
            
            total_length = compute_total_cost(child_ls, instance["edge_weight"])

            # Check the capacity constraint
            feasibility = capacity_check(child_ls, instance)
            if feasibility is True:
                if total_length < best_cost:
                    best_cost = total_length
                    best_sol = child_ls
                    new_best_sol_found = True
            else:
                total_length = penalty*total_length # Penalty approach for infeasible solutions
            # Update population
            pop.append({"cromosoms":[node for route in new_routes for node in route], # Encoding of the child educated with LS
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
        it +=1
        if it%50 == 0:
            it_ls += 1 # As the algorithm proceeds we increase the iterations of LS

    return best_sol
    