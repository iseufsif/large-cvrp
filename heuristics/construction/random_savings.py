import random

def randomized_savings(instance, alpha=0.3):
    """
    Generates a random solution for VRP starting from the savings algorithm
    In each iteration, it selects a random saving among the top #alpha and creates a route
    with that couple of nodes

    Not yet working as intended, thus excluded from the benchmark

    Return:
        List[List[int]]: random solution for VRP
    """
    n = len(instance["demand"])
    demands = instance["demand"]
    capacity = instance["capacity"]
    edge_weight = instance["edge_weight"]

    customers = list(range(1, n))  # exclude depot

    # Step 1: compute savings
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                saving = edge_weight[0][i] + edge_weight[0][j] - edge_weight[i][j]
                savings.append((saving, i, j))
    savings.sort(reverse=True)

    # Step 2: initialize one route per customer
    routes = [[i] for i in customers]
    customer_to_route = {i: r for r in routes for i in r}

    # Step 3: merge routes based on savings
    while savings:
        k = max(1, int(alpha * len(savings)))
        _, i, j = random.choice(savings[:k])
        savings = [s for s in savings if i not in s[1:] and j not in s[1:]]

        r1 = customer_to_route.get(i)
        r2 = customer_to_route.get(j)

        if r1 is None or r2 is None or r1 == r2:
            continue

        # merge if feasible and endpoints match
        if r1[-1] == i and r2[0] == j:
            merged = r1 + r2
        elif r2[-1] == i and r1[0] == j:
            merged = r2 + r1
        else:
            continue

        if sum(demands[c] for c in merged) <= capacity:
            # remove old routes
            routes.remove(r1)
            routes.remove(r2)
            routes.append(merged)

            for c in merged:
                customer_to_route[c] = merged

    # Integrity check
    all_customers = set(c for r in routes for c in r)
    expected = set(range(1, n))
    if all_customers != expected:
        missing = expected - all_customers
        extra = all_customers - expected
        raise RuntimeError(f"Final savings routes inconsistent: Missing {missing}, Extra {extra}")

    return routes
