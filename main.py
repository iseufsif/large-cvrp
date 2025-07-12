import random

from heuristics.construction.random import generate_random_solution
from utils.utils import compute_total_cost, write_solution, print_solution, log_results
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls_with_2opt, ls_with_swaps
from heuristics.metaheuristics.diversifying_components.simulated_annealing import simulated_annealing
from heuristics.metaheuristics.instensifying_components.tabu import tabu_search
from heuristics.metaheuristics.diversifying_components.genetic_algorithm import genetic_algorithm
from heuristics.metaheuristics.diversifying_components.hybrid_genetic_search import HGS
from heuristics.metaheuristics.diversifying_components.lns import lns


def main():
    history = []

    random.seed(42)
    # Examplary Instance
    instance_name = "A-n45-k6.vrp"

    # Generate random initial solution and initialize instance
    random_routes, instance = generate_random_solution(instance_name)
    log_results("Random Solution", random_routes, instance, history)
    
    # Standalone Local Search - Swap-based
    ls_swap_routes = ls_with_swaps(instance, random_routes, 1000)
    log_results("Local Search (Swaps)", ls_swap_routes, instance, history)

    # Standalone Local Search - 2-Opt
    ls_2opt_routes = ls_with_2opt(instance, random_routes, 1000)
    log_results("Local Search (2-Opt)", ls_2opt_routes, instance, history)

    # Local Search Pipeline - Swap -> 2-Opt
    ls_pipeline_routes = ls_with_2opt(instance, ls_swap_routes, 1000)
    log_results("Local Search Pipeline", ls_pipeline_routes, instance, history)

    # Standalone Simulated Annealing
    sa_routes = simulated_annealing(instance, random_routes)
    log_results("Standalone SA", sa_routes, instance, history)

    # LS + Simulated Annealing
    ls_sa_routes = simulated_annealing(instance, ls_pipeline_routes)
    log_results("LS Pipeline + SA", ls_sa_routes, instance, history)

    # Tabu Search
    tabu_routes = tabu_search(instance, random_routes, 500)
    log_results("Standalone Tabu Search", tabu_routes, instance, history)

    # LNS
    lns_routes = lns(instance, random_routes, 1000)
    log_results("Standalone LNS", lns_routes, instance, history)


    """# Random initial population for GA
    initial_population = []
    pop_size = 4
    for i in range(pop_size):
        routes, instance = generate_random_solution(instance_name)
        initial_population.append(routes)
        cost = compute_total_cost(routes, instance["edge_weight"])
        print(f"Solution {i} in initial population = {routes},\nCost = {cost}")
    print("---------------------------")
    
    # Call GA
    print("--- Solution GA ---")
    routes_ga = HGS(initial_population, instance, pop_size)
    cost_ga = compute_total_cost(routes_ga, instance["edge_weight"])
    print_solution(routes_ga, cost_ga)
    print("---------------------------")"""

    # Print overview of all results
    print("\n======= Heuristic Comparison =======")
    for label, cost, _ in history:
        print(f"{label:<25} | Cost: {cost}")

    # Identify best solution from history and plot this one
    best_label, best_cost, best_routes = min(history, key=lambda x: x[1])
    plot_routes(instance, best_routes, title=f"Best: {best_label} (Cost: {best_cost})")

if __name__ == "__main__":
    main()