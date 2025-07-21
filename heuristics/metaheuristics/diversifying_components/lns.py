from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.remove import random_removal, worst_removal
from heuristics.metaheuristics.neighborhood_operators.repair import greedy_repair, regret_repair


def fast_lns(instance, routes, min_iter=250, destroy_frac=0.1): # destroy_frac finetuned
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

def smart_lns(instance, routes, min_iter=100, destroy_frac=0.4): # destroy_frac finetuned
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