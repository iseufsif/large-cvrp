import random
from utils.utils import compute_total_cost, is_feasible

def randomized_savings(instance, alpha=0.3):
    n = len(instance["demand"])
    demands = instance["demand"]
    capacity = instance["capacity"]
    edge_weight = instance["edge_weight"]

    customers = list(range(1, n))  # exclude depot (0)

    # Step 1: compute savings list
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                saving = edge_weight[0][i] + edge_weight[0][j] - edge_weight[i][j]
                savings.append((saving, i, j))

    # Step 2: sort by savings descending
    savings.sort(reverse=True)

    # Step 3: initialize one route per customer
    routes = [[i] for i in customers]

    # Map each customer to its route
    cust_to_route = {i: route for route in routes for i in route}

    # Step 4: merge routes based on savings (randomized)
    while savings:
        # Select from top-k% of sorted savings
        k = max(1, int(alpha * len(savings)))
        saving, i, j = random.choice(savings[:k])
        savings = [s for s in savings if s[1] != i and s[2] != j and s[1] != j and s[2] != i]

        r1 = cust_to_route.get(i)
        r2 = cust_to_route.get(j)

        if r1 is None or r2 is None or r1 == r2:
            continue

        # Check for feasibility and that i is at the end of r1, j at start of r2 (or vice versa)
        if r1[-1] == i and r2[0] == j:
            new_route = r1 + r2
        elif r2[-1] == i and r1[0] == j:
            new_route = r2 + r1
        else:
            continue

        if sum(demands[c] for c in new_route) <= capacity:
            # Merge routes
            routes.remove(r1)
            routes.remove(r2)
            routes.append(new_route)
            for c in new_route:
                cust_to_route[c] = new_route

    return routes
