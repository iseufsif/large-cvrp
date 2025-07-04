import vrplib # to load the instances
import random

from heuristics.construction.random import generate_random_solution
from utils.utils import compute_total_cost, write_solution, print_solution
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls

def main():
    random.seed(42)
    # Examplary Instance
    instance_name = "A-n45-k6.vrp"

    # Generate random solution
    routes, instance = generate_random_solution(instance_name)
    cost = compute_total_cost(routes, instance["edge_weight"])
    print_solution(routes, cost)

    # Call LS
    routes_ls = ls(instance, routes, 1000)
    cost_ls = compute_total_cost(routes_ls, instance["edge_weight"])
    print_solution(routes_ls, cost_ls)

    # Plot final result
    plot_routes(instance, routes_ls, title=f"Random Solution for {instance_name}")

if __name__ == "__main__":
    main()