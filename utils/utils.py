import os

def compute_total_cost(routes, edge_weight):
    total_cost = 0
    for route in routes:
        prev = 0  # start at depot (index 0)
        for cust in route:
            total_cost += edge_weight[prev][cust]
            prev = cust
        total_cost += edge_weight[prev][0]  # return to depot
    return total_cost

def write_solution(file_path, routes, cost):
    with open(file_path, "w") as f:
        for i, route in enumerate(routes, 1):
            f.write(f"Route #{i}: {' '.join(str(c) for c in route)}\n")
        f.write(f"Cost {int(round(cost))}\n")

def print_solution(routes, cost):
    for i, route in enumerate(routes, 1):
        print(f"Route #{i}: {' '.join(str(c) for c in route)}")
    print(f"Cost {int(round(cost))}")

def is_feasible(route, instance):
    total_demand = sum(instance["demand"][cust] for cust in route)
    return total_demand <= instance["capacity"]

def log_results(label, routes, instance, history, runtime=None, bks=None):
    cost = round(compute_total_cost(routes, instance["edge_weight"]), 4)
    assert all(is_feasible(r, instance) for r in routes), \
        f"Infeasible solution detected in {label}"
    # Calculate the gap in percent between our solutions and the best known solution (bks)
    gap = ((cost - bks) / bks * 100) if bks else None
    history.append((label, cost, gap, runtime, routes))
    # print(f"{label:<25} | Cost: {cost}")

def get_bks(instance_name):
    sol_file = os.path.join("solutions", instance_name.replace(".vrp", ".sol"))
    with open(sol_file, "r") as f:
        for line in f:
            if line.startswith("Cost"):
                return float(line.split()[1])
    raise ValueError(f"No cost line found in {sol_file}")
