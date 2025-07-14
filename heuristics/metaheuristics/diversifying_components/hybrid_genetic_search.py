from heuristics.metaheuristics.diversifying_components.genetic_algorithm import fitness, calculate_probabilities, parent_selection, encoding, decoding, uniform_crossover, capacity_check, mutation
from utils.tsp_solvers_for_GA import tsp_solver_nn, tsp_solver_ls
from heuristics.metaheuristics.instensifying_components.ls import hybrid_ls
import numpy as np
from utils.utils import compute_total_cost
import random
import math

def order_crossover(parent1, parent2):
    size = len(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    current_index = (end + 1) % size
    parent2_index = (end + 1) % size

    while None in child:
        gene = parent2[parent2_index]
        child[current_index] = gene
        current_index = (current_index + 1) % size
        parent2_index = (parent2_index + 1) % size

    return child

def HGS(initial_population, instance, pop_size, max_no_improv = 50):
    # Initialize the population of individuals
    pop = [] 
    for sol in initial_population:
        ind = encoding(sol, instance)
        pop.append(ind)
    fitness(pop)
    calculate_probabilities(pop)
    
    gen_size = 25
    penalty = 3
    ref_ratio = 0.2
    n_elite = 4
    no_improv = 0
    it = 1
    it_ls = 0
    best_cost = min(ind["Z"] for ind in pop)

    # Algorithm
    while no_improv < max_no_improv: 
        new_best_sol_found = False
        fitness(pop)
        calculate_probabilities(pop)
        #print(f"\nIteration {it}")
        for _ in range(gen_size):
            # Parent Selection
            pop_sorted = sorted(pop, key=lambda x: x["Z"])
            elite = pop_sorted[:n_elite] # First Parent among the top n_elite solutions 
            diverse = pop_sorted[n_elite:] 

            parent_1 = parent_selection(elite)
            parent_2 = parent_selection(diverse)
    
            # Reprodcution
            child = uniform_crossover(parent_1, parent_2)
            child_decoded = decoding(child)
            # Educate child using Nearest Neighbor and Local Search
            new_routes = [] 
            total_length = 0
            for route in child_decoded:
                sequenced_route, route_length = tsp_solver_nn(route, instance["edge_weight"])
                new_routes.append(sequenced_route)  

            estimated_cost = compute_total_cost(new_routes, instance["edge_weight"])
            if estimated_cost > 1.5 *best_cost:
                #print("SKIPPED")
                continue  # skip LS on bad offspring
            #elif estimated_cost < 1.1 * best_cost:
               # print("APPEALING")
                #child_ls = hybrid_ls(instance, new_routes, 20) # Deeper search for appealing solutions
            else:
                #print("NORMAL")
                child_ls = hybrid_ls(instance, new_routes, min(10, 3 + it_ls))
            
            total_length = compute_total_cost(child_ls, instance["edge_weight"])
            # Check the capacity constraint
            feasibility = capacity_check(child_ls, instance)
            if feasibility is True:
                #print("Cost New Feasible Solution=", total_length)
                if total_length < best_cost:
                    best_cost = total_length
                    best_sol = child_ls
                    new_best_sol_found = True
                    #print(f"In Iteration {it}: Improved to cost {best_cost:.2f}")
            else:
                total_length = penalty*total_length # Penalty approach for infeasible solutions
            # Update population
            pop.append({"cromosoms":child, "Z": total_length, "f":0, "p":0, "range":[0,1], "feasible": feasibility})

        # Replacement
        pop = sorted(pop, key=lambda ind: ind["Z"])[:pop_size+gen_size]
        #print(f"In iteration {it}, population = {pop}")
        num_infeasible = 0
        num_feasible = 0

        # Population Management
        for sol in pop:
            if sol["feasible"] is True:
                num_feasible += 1
            else:
                num_infeasible += 1 

        ratio = num_infeasible/num_feasible

        if ratio > ref_ratio: # We want to have 20% of infeasible solutions in our population
            penalty = penalty*1.1
        else:
            penalty = penalty*0.9
        
        # Improvement Management
        if new_best_sol_found is True:
            no_improv = 0
        else:
            no_improv += 1
        print(no_improv)
        it +=1
        if it%10 == 0:
            it_ls += 1

    return best_sol
    