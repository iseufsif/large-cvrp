from heuristics.metaheuristics.neighborhood_operators.remove import random_removal
from heuristics.metaheuristics.neighborhood_operators.repair import greedy_repair
from heuristics.metaheuristics.instensifying_components.ls import hybrid_ls
from utils.utils import compute_total_cost

def perturb(instance, routes, destroy_factor=0.1):
    num_customers = sum(len(r) for r in routes)
    num_remove = int(num_customers * destroy_factor)

    partial, removed = random_removal(routes, num_remove)
    repaired = greedy_repair(instance, partial, removed)
    return repaired

def iterated_local_search(instance, initial_solution, ls=hybrid_ls, it=100, destroy_factor=0.1):
    current = ls(instance, initial_solution, it)
    current_cost = compute_total_cost(current, instance["edge_weight"])
    best = current
    best_cost = current_cost

    for _ in range(it):
        perturbed = perturb(instance, current, destroy_factor)
        improved = ls(instance, perturbed)
        improved_cost = compute_total_cost(improved, instance["edge_weight"])

        if improved_cost < current_cost:
            current = improved
            current_cost = improved_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

    return best
