import vrplib # to load the instances
import random

from heuristics.construction.random import generate_random_solution
from utils.utils import compute_total_cost, write_solution, print_solution
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls_with_2opt, ls_with_swaps

def main():
    random.seed(42)
    # Examplary Instance
    instance_name = "A-n45-k6.vrp"

    # Generate random solution
    routes, instance = generate_random_solution(instance_name)
    cost = compute_total_cost(routes, instance["edge_weight"])
    print_solution(routes, cost)

    # Call LS with Swaps
    routes_ls = ls_with_swaps(instance, routes, 1000)
    cost_ls = compute_total_cost(routes_ls, instance["edge_weight"])
    print_solution(routes_ls, cost_ls)

    # Call LS with 2-Opt
    routes_ls_2 = ls_with_2opt(instance, routes_ls, 1000)
    cost_ls_2 = compute_total_cost(routes_ls_2, instance["edge_weight"])
    print_solution(routes_ls_2, cost_ls_2)

    # Plot final result
    plot_routes(instance, routes_ls_2, title=f"Random Solution for {instance_name}")

if __name__ == "__main__":
    main()