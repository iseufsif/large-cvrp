import vrplib # to load the instances
import random

from heuristics.random import generate_random_solution
from utils.utils import compute_total_cost, write_solution, print_solution
from utils.plot import plot_routes

def main():
    # Examplary Instance
    instance_name = "A-n45-k6.vrp"

    # Generate random solution
    routes, instance = generate_random_solution(instance_name)

    # Calculate cost
    cost = compute_total_cost(routes, instance["edge_weight"])

    print_solution(routes, cost)

    plot_routes(instance, routes, title=f"Random Solution for {instance_name}")

if __name__ == "__main__":
    main()