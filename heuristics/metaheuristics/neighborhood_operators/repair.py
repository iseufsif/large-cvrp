from utils.utils import is_feasible, compute_total_cost

# greedy repair is the cheapest feasible insertion
def greedy_repair(instance, solution, removed_customers):
    for cust in removed_customers:
        best_cost = float('inf')
        best_position = None
        best_route_idx = None

        for r_idx, route in enumerate(solution):
            for i in range(len(route) + 1):
                new_route = route[:i] + [cust] + route[i:]
                if is_feasible(new_route, instance):
                    # Compute cost with depot (0) at start and end
                    cost = compute_total_cost([new_route], instance["edge_weight"])
                    if cost < best_cost:
                        best_cost = cost
                        best_position = i
                        best_route_idx = r_idx

        if best_position is not None:
            solution[best_route_idx].insert(best_position, cust)
        else:
            # Start new route if all others violate capacity
            solution.append([cust])

    return solution
