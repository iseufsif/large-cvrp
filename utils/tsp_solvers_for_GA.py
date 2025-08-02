import random
import copy

def tsp_solver_nn(route, distances):
    """
    Solves TSP using Nearest Neighbor

    Return:
        List[int]: Solution of TSP using NN
        Int: Total length of the solution
    """
    if len(route) == 0: # Check for children born with a lower number of vehicles
        return route, 0
    tour = []
    not_visited = copy.copy(route)
    prev = 0
    iteration = 1
    length = 0
    while len(not_visited) > 0:
        sorted_remaining = sorted(not_visited, key=lambda node: distances[prev, node]) 
        current_node = sorted_remaining[0]
        tour.append(current_node)
        length += distances[prev, current_node] # update length
        not_visited.remove(current_node) # update not visited
        prev = current_node
        iteration += 1
    length += distances[current_node, 0]
    return tour, length

def tsp_solver_ls(route, distances, it = 5):
    """
    Solves TSP Local Search
    Neighborhood Operator: Two_Opt

    Return:
        List[int]: Solution of TSP using LS
        Int: Total length of the solution
    """
    current_sol = route
    current_length = evaluate_TSP_sol(route, distances)
    # Solve TSP with Descent Local Search
    improv = True
    while improv:
        improv = False
        for i in range(it):
            if len(route) == 3:
                index_1 = random.choice([0,2])
                index_2 = 1
            elif len(route) <= 2:
                break
            else:
                index_1 = random.randint(1,len(current_sol)-2) 
                index_2 = random.randint(index_1, len(current_sol)-1)
            neighbor = two_opt_move(current_sol, index_1, index_2)
            neighbor_length = evaluate_TSP_sol(neighbor, distances)
            delta = neighbor_length - current_length
            if delta < 0:
                current_sol = neighbor.copy()       
                current_length = neighbor_length    
                improv = True
                break        
    return current_sol, current_length

def two_opt_move(tour, i, j): # Function to define the new tour with 2-opt given the nodes to swap
    """Perform a 2-opt move by reversing the segment between indices i and j."""
    if len(tour) >= 4:
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    else:
        new_tour = copy.copy(tour)
        new_tour[j] = tour[i]
        new_tour[i] = tour[j]
        return new_tour

def evaluate_TSP_sol(route, edge_weight):
    """Compute total length of a given TSP solution"""
    total_cost = 0
    prev = 0  # start at depot (index 0)
    for cust in route:
        total_cost += edge_weight[prev][cust]
        prev = cust
    total_cost += edge_weight[prev][0]  # return to depot
    return total_cost