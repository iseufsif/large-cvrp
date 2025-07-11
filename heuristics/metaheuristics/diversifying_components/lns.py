from utils.utils import compute_total_cost
from heuristics.metaheuristics.neighborhood_operators.remove import random_removal
from heuristics.metaheuristics.neighborhood_operators.repair import greedy_repair


def lns(instance, routes, max_iter=250, destroy_frac=0.2):
    current_solution = routes
    best_solution = current_solution
    best_cost = compute_total_cost(current_solution, instance["edge_weight"])

    for it in range(max_iter):
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

        # if it % 25 == 0:
        #     print(f"Iter {it}: cost = {cost}, best = {best_cost}")

    return best_solution
