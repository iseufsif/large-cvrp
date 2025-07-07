import vrplib # to load the instances
import random

from heuristics.construction.random import generate_random_solution
from utils.utils import compute_total_cost, write_solution, print_solution
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls_with_2opt, ls_with_swaps
from heuristics.metaheuristics.diversifying_components.simulated_annealing import simulated_annealing
from heuristics.metaheuristics.instensifying_components.tabu import tabu_search
from heuristics.metaheuristics.diversifying_components.genetic_algorithm import genetic_algorithm

def main():
    random.seed(42)
    # Examplary Instance
    instance_name = "A-n45-k6.vrp"
    
    """# Random initial population for GA
    initial_population = []
    for i in range(10):
        routes, instance = generate_random_solution(instance_name)
        initial_population.append(routes)
        cost = compute_total_cost(routes, instance["edge_weight"])
        print(f"Solution {i} in initial population = {routes},\nCost = {cost}")
    print("---------------------------")
    
    # Call GA
    print("--- Solution GA ---")
    routes_ga = genetic_algorithm(initial_population, instance)
    cost_ga = compute_total_cost(routes_ga, instance["edge_weight"])
    print_solution(routes_ga, cost_ga)
    print("---------------------------")"""


    # Generate random solution
    print("--- Randomized Solution ---")
    routes, instance = generate_random_solution(instance_name)
    cost = compute_total_cost(routes, instance["edge_weight"])
    print_solution(routes, cost)
    print("---------------------------")

    # Call LS with Swaps
    print("--- Solution LS #1 ---")
    routes_ls = ls_with_swaps(instance, routes, 1000)
    cost_ls = compute_total_cost(routes_ls, instance["edge_weight"])
    print_solution(routes_ls, cost_ls)
    print("----------------------")

    # Call LS with 2-Opt
    print("--- Solution LS #2 ---")
    routes_ls_2 = ls_with_2opt(instance, routes_ls, 1000)
    cost_ls_2 = compute_total_cost(routes_ls_2, instance["edge_weight"])
    print_solution(routes_ls_2, cost_ls_2)
    print("----------------------")

    # Call Simulated Annealing
    print("--- Solution SA ---")
    routes_sa = simulated_annealing(instance, routes_ls_2)
    cost_sa = compute_total_cost(routes_sa, instance["edge_weight"])
    print_solution(routes_sa, cost_sa)
    print("-------------------")

    # Call Tabu Search
    print("--- Solution Tabu Search ---")
    routes_tabu = tabu_search(instance, routes_sa, 500)
    cost_tabu = compute_total_cost(routes_tabu, instance["edge_weight"])
    print_solution(routes_tabu, cost_tabu)
    print("----------------------")

    # Plot final result

    plot_routes(instance, routes_tabu, title=f"Random Solution for {instance_name}")

if __name__ == "__main__":
    main()