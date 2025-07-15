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


def regret_repair(instance, solution, removed_customers, k=3):
    while removed_customers:
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