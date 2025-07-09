from heuristics.metaheuristics.diversifying_components.genetic_algorithm import fitness, calculate_probabilities, parent_selection, encoding, decoding, uniform_crossover, capacity_check
from utils.tsp_solvers_for_GA import tsp_solver_ls, tsp_solver_nn
from heuristics.metaheuristics.instensifying_components.ls import ls_with_swaps
import numpy as np
from utils.utils import compute_total_cost

def HGS(initial_population, instance, pop_size, max_no_improv = 100):
    # Initialize the population of individuals
    pop = [] 
    for sol in initial_population:
        ind = encoding(sol, instance)
        pop.append(ind)
    fitness(pop)
    calculate_probabilities(pop)

    p_rek = 0.7
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
    
            child_decoded = decoding(child)
            print("\nNew Child")
            # Educate child using Local Search
            new_routes = [] 
            total_length = 0
            for route in child_decoded:
                sequenced_route, route_length = tsp_solver_nn(route, instance["edge_weight"])
                new_routes.append(sequenced_route)  
                
            child_ls = ls_with_swaps(instance, new_routes, 10)
            total_length = compute_total_cost(child_ls, instance["edge_weight"])
            # Check the capacity constraint
            feasibility = capacity_check(child_ls, instance)
            if feasibility is True:
                print("Cost New Feasible Solution=", total_length)
            else:
                total_length += 10*total_length # Penalty approach for infeasible solutions
            # Update population
            pop.append({"cromosoms":child, "Z": total_length, "f":0, "p":0, "range":[0,1], "feasible": feasibility})

        # Replacement
        pop = sorted(pop, key=lambda ind: ind["Z"])[:pop_size] 
        if round(pop[0]["Z"]) < round(best_cost):
            best_cost = pop[0]["Z"]
            best_sol = pop[0]["cromosoms"]
            no_improv = 0
            print(f"In iteration {it}, population = {pop}")
            print(f"In Iteration {it}: Improved to cost {best_cost:.2f}")
        else:
            no_improv += 1
        it +=1

    best_sol_decoded = decoding(best_sol)
    best_sol = ls_with_swaps(instance, best_sol_decoded, 100)
    
    return best_sol
    