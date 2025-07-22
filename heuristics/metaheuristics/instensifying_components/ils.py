from heuristics.metaheuristics.neighborhood_operators.remove import random_removal
from heuristics.metaheuristics.neighborhood_operators.repair import greedy_repair
from heuristics.improvement.ls import hybrid_ls
from utils.utils import compute_total_cost
import copy
import random

def perturb(instance, routes, destroy_factor=0.2):
    num_customers = sum(len(r) for r in routes)
    num_remove = int(num_customers * destroy_factor)

    partial, removed = random_removal(routes, num_remove)
    repaired = greedy_repair(instance, partial, removed)
    # if repaired is None:
    #     raise RuntimeError("❌ [ILS] Repair failed and returned None")
    return repaired

def iterated_local_search(instance, initial_solution, ls=hybrid_ls, it=100, destroy_factor=0.2): # destroy_factor finetuned

    current = ls(instance, copy.deepcopy(initial_solution), it)
    current_cost = compute_total_cost(current, instance["edge_weight"])
    best = current
    best_cost = current_cost

    # this helps to start ls with fewer iterations in early stages of ILS, where gains are found quickly
    base_ils_it = int(it / 5)
    max_ils_it = max(it, instance["dimension"])

    for _ in range(it):
        perturbed = perturb(instance, current, destroy_factor)
        # check_integrity(perturbed, instance)
        adaptive_it = int(base_ils_it + (max_ils_it - base_ils_it) * (_ / it))
        improved = ls(instance, perturbed, it=adaptive_it)
        improved_cost = compute_total_cost(improved, instance["edge_weight"])

        if improved_cost < current_cost:
            current = improved
            current_cost = improved_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost

    return best


### Functions for Debugging Purposes ###

def check_integrity(routes, instance):
    expected = set(range(1, instance["dimension"]))
    visited = set(c for route in routes for c in route)
    missing = expected - visited
    extra = visited - expected
    if missing:
        raise ValueError(f"❌ Missing customers: {missing}")
    if extra:
        raise ValueError(f"❌ Unexpected customers: {extra}")
    
def debug_check_perturb(instance, partial, removed):
    removed_set = set(removed)
    partial_set = set(c for r in partial for c in r)
    total_set = removed_set | partial_set
    expected = set(range(1, instance["dimension"]))
    missing = expected - total_set
    if missing:
        print("❌ Customers lost in perturbation before repair:", missing)
        print("Removed:", sorted(removed_set))
        print("Remaining in partial:", sorted(partial_set))
        print("Expected:", sorted(expected))
        raise RuntimeError("❌ Bug in random_removal: customers got dropped")

def check_solution_integrity(routes, instance, label):
    expected = set(range(1, instance["dimension"]))
    visited = [c for r in routes for c in r]
    visited_set = set(visited)
    missing = expected - visited_set
    duplicates = set(c for c in visited if visited.count(c) > 1)

    if missing or duplicates:
        print(f"❌ [{label}] Customer integrity issue:")
        if missing:
            print(f"   → Missing: {sorted(missing)}")
        if duplicates:
            print(f"   → Duplicates: {sorted(duplicates)}")
        raise RuntimeError(f"❌ [{label}] Invalid input solution")


