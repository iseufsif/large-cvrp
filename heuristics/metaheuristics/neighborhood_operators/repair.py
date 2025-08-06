import copy
from utils.utils import is_feasible, compute_total_cost
import random

# greedy repair is the cheapest feasible insertion
def greedy_repair(instance, solution, removed_customers):
    """
    Performs greedy insertion to repair a VRP solution

    Return:
    List[List[int]]: repaired VRP solution
    """
    solution = copy.deepcopy(solution)
    inserted_customers = set()

    for cust in removed_customers:
        best_cost = float('inf')
        best_position = None
        best_route_idx = None

        # this helps go go from O(n**2) to O(n) -- tradeoff: performance (-) for runtime (-)
        sampled_routes = random.sample(solution, min(len(solution), 25))  # sample 25 routes

        for route in sampled_routes:
            r_idx = solution.index(route)
            for i in range(len(route) + 1):
                prev = route[i - 1] if i > 0 else 0
                next = route[i] if i < len(route) else 0

                delta_cost = (
                    instance["edge_weight"][prev][cust] +
                    instance["edge_weight"][cust][next] -
                    instance["edge_weight"][prev][next]
                )

                if delta_cost < best_cost:
                    new_route = route[:i] + [cust] + route[i:]
                    if is_feasible(new_route, instance):
                        best_cost = delta_cost
                        best_position = i
                        best_route_idx = r_idx

        if best_position is not None:
            solution[best_route_idx].insert(best_position, cust)
            inserted_customers.add(cust)
        else:
            solution.append([cust])
            inserted_customers.add(cust)

    # Final integrity check with detailed output
    all_customers = set(range(1, instance["dimension"]))
    visited = set(c for route in solution for c in route)

    missing = all_customers - visited
    extra   = visited - all_customers

    if missing or extra:
        print("Final inserted_customers:", sorted(inserted_customers))
        print("Final visited customers:", sorted(visited))
        print("Expected customers     :", sorted(all_customers))
        if missing:
            print("Still missing customers:", missing)
        if extra:
            print("Unexpected extra customers:", extra)
        raise RuntimeError("[greedy_repair] Final customer mismatch after repair")

    return solution

def regret_repair(instance, solution, removed_customers, k=3):
    """
    Performs regret K insertion to repair a VRP solution

    Return:
    List[List[int]]: repaired VRP solution
    """
    while removed_customers:
        solution = copy.deepcopy(solution)
        regret_list = []

        for cust in removed_customers:
            insertions = []
            for r_idx, route in enumerate(solution):
                for i in range(len(route) + 1):
                    new_route = route[:i] + [cust] + route[i:]
                    if is_feasible(new_route, instance):
                        cost = compute_total_cost([new_route], instance["edge_weight"])
                        insertions.append((cost, r_idx, i))
            
            if insertions:
                insertions.sort()
                if len(insertions) >= k:
                    regret = insertions[k-1][0] - insertions[0][0]
                else:
                    regret = insertions[-1][0] - insertions[0][0]

                best_cost, best_r_idx, best_pos = insertions[0]
                regret_list.append((regret, best_cost, cust, best_r_idx, best_pos))
            else:
                # No feasible insertions for now → will be appended to new route
                regret_list.append((-1, float('inf'), cust, None, None))

        # Choose customer with highest regret
        regret_list.sort(reverse=True)
        _, _, chosen_cust, r_idx, pos = regret_list[0]
        removed_customers.remove(chosen_cust)

        if r_idx is not None:
            solution[r_idx].insert(pos, chosen_cust)
        else:
            solution.append([chosen_cust])

    return solution