from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.remove import random_removal, worst_removal
from heuristics.metaheuristics.neighborhood_operators.repair import greedy_repair, regret_repair


def fast_lns(instance, routes, min_iter=250, destroy_frac=0.1): # destroy_frac finetuned
    """
    Performs Fast Large Neighborhood Search for VRP

    Fast refers to the adopted destroy and repair operators:
    - Destroy Operator: random removal
    - Repair Operator: greedy repair

    Return:
        List[List[int]]: best found solution for the VRP problem
    """
    current_solution = routes
    best_solution = current_solution
    best_cost = compute_total_cost(current_solution, instance["edge_weight"])

    iterations = max(min_iter, instance["dimension"])

    for it in range(iterations):
        # destroy initial solution
        num_remove = int(destroy_frac * sum(len(r) for r in current_solution))
        partial_solution, removed = random_removal(current_solution, num_remove)

        # repair destroyed solution
        repaired = greedy_repair(instance, partial_solution, removed)
        cost = compute_total_cost(repaired, instance["edge_weight"])

        if cost < best_cost:
            best_solution = repaired
            best_cost = cost
            current_solution = repaired

    return best_solution

def smart_lns(instance, routes, min_iter=250, destroy_frac=0.1): # destroy_frac finetuned
    """
    Performs Smart Large Neighborhood Search for VRP
    
    Smart refers to the adopted destroy and repair operators:
    - Destroy Operator: worst removal
    - Repair Operator: regret repair

    Return:
        List[List[int]]: best found solution for the VRP problem
    """
    current_solution = routes
    best_solution = current_solution
    best_cost = compute_total_cost(current_solution, instance["edge_weight"])

    iterations = max(min_iter, instance["dimension"])

    for it in range(iterations):
        # destroy initial solution
        num_remove = int(destroy_frac * sum(len(r) for r in current_solution))
        partial_solution, removed = worst_removal(instance, current_solution, num_remove)

        # repair destroyed solution
        repaired = regret_repair(instance, partial_solution, removed)
        cost = compute_total_cost(repaired, instance["edge_weight"])

        if cost < best_cost:
            best_solution = repaired
            best_cost = cost
            current_solution = repaired

    return best_solution