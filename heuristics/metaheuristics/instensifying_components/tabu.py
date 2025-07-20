from utils.utils import compute_total_cost
import random
import math
import copy
from heuristics.metaheuristics.neighborhood_operators.exchange import exchange_move


def tabu_search(instance, routes, it = 100):
    current_sol = copy.deepcopy(routes) # A feasible solution to the minimization problem: x
    current_best_sol = copy.deepcopy(routes) # Initial best solution x*
    current_length = compute_total_cost(current_sol, instance["edge_weight"]) # Initial F(x)   
    current_best_length = compute_total_cost(current_best_sol, instance["edge_weight"]) # Initial F(x*)

    tabu_list = []
    n = instance["dimension"]
    
    # Parameters:
    max_length_tabu = math.sqrt(n) # tuned
    size_neighborhood = round(0.8*n) # tuned

    for k in range(it):
        best_delta = 0
        best_worse_delta = sum(sum(instance["edge_weight"]))
        best_solution = current_best_sol.copy()
        best_worse_neighbor = current_sol.copy() 
        to_be_added = None

        # Neighborhood: 30 two-opt moves for random nodes of the current solution
        for iter in range(size_neighborhood):
            route_1_idx = random.randint(0,len(current_sol)-1)
            route_2_idx = random.randint(0,len(current_sol)-1)
            # Check if the selected move is in the tabu list
            if [route_1_idx, route_2_idx] not in tabu_list and [route_2_idx, route_1_idx] not in tabu_list:
                neighbor, delta = exchange_move(current_sol, route_1_idx, route_2_idx, instance)
                neighbor_length = compute_total_cost(neighbor, instance["edge_weight"])

                # Check if the new neighbor is a new best solution
                if neighbor_length >= current_best_length:
                    # If not we calculate the delta with the current solution
                    delta = neighbor_length - current_length 
                    # If it is the neighbor with the lowest F(x') [and F(x') < F(x*) previously checked] we select it
                    if delta <= best_worse_delta:
                        best_worse_delta = delta
                        best_worse_neighbor = neighbor.copy()
                        to_be_added = [route_1_idx, route_2_idx] # we will add this move to the tabo list

                # if we found a new best solution we save it
                else: 
                    delta_best = neighbor_length - current_best_length
                    if delta_best < best_delta:
                        best_solution = neighbor.copy()
                        best_delta = delta_best
                        # print(f"In Iteration {k}: Improved to cost {current_best_length + best_delta:.2f}")
        # Update the tabo list:
        if to_be_added != None:         
            tabu_list.append(to_be_added) 

        if len(tabu_list) > max_length_tabu: 
            tabu_list.pop(0)

        current_sol = best_worse_neighbor # Update the current solution x <- x'
        current_length += best_worse_delta # Update current length F(x) <- F(x')
        current_best_sol = best_solution # In case we found an improvement x* <- x    
        current_best_length += best_delta  # Update F(x*)
    
    return current_best_sol