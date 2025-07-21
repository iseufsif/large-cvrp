import random
import time
import copy
import math

from multiprocessing import Pool, cpu_count
from heuristics.construction.random import generate_random_solution
from heuristics.construction.random_savings import randomized_savings
from utils.utils import compute_total_cost, write_solution, print_solution, log_results, get_bks
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls_with_2opt, ls_with_swaps, hybrid_ls
from heuristics.metaheuristics.instensifying_components.ils import iterated_local_search, check_solution_integrity
from heuristics.metaheuristics.diversifying_components.simulated_annealing import simulated_annealing
from heuristics.metaheuristics.instensifying_components.tabu import tabu_search
from heuristics.metaheuristics.diversifying_components.genetic_algorithm import genetic_algorithm
from heuristics.metaheuristics.diversifying_components.hybrid_genetic_search import HGS
from heuristics.metaheuristics.diversifying_components.lns import fast_lns, smart_lns
from utils.tsp_solvers_for_GA import tsp_solver_nn

def run_tabu_experiment(args):
    """
    This is mulitprocessing for finetuning purposes. Edit the args you want to tune and
    adjust the rest of the code accordingly.

    Then post the following code snippet in the main function:

        n = instance["dimension"]
        tabu_lens = [int(math.sqrt(n)), int(0.05 * n), int(0.1 * n), int(math.log(n))]
        neigh_fracs = [0.2, 0.5, 0.8, 1.0]
        args_list = [(tl, nf, instance, random_routes, bks) for tl in tabu_lens for nf in neigh_fracs]

        with Pool(processes=min(cpu_count(), len(args_list))) as pool:
            results = pool.map(run_tabu_experiment, args_list)

        for res in results:
            log_results(res["label"], res["routes"], instance, history, runtime=res["elapsed"], bks=bks)

    Adjust the parameters you want to vary with values to create a sweep grid or simple parameter variation.
    In the example above we vary tabu list length and neighborhood fraction for tabu search.
    """

    tabu_len, neigh_frac, instance, base_routes, bks = args
    import time
    n = instance["dimension"]
    size_neighborhood = int(neigh_frac * n)
    max_length_tabu = tabu_len

    start = time.time()
    solution = tabu_search(instance, copy.deepcopy(base_routes), max_no_improv=100, 
                           size_neighborhood=size_neighborhood,
                           max_length_tabu=max_length_tabu)
    elapsed = round((time.time() - start) / 60, 4)
    cost = compute_total_cost(solution, instance["edge_weight"])
    return {
        "label": f"Tabu (tabu={tabu_len}, neigh={neigh_frac})",
        "cost": cost,
        "elapsed": elapsed,
        "routes": solution
    }


def main():
    total_start = time.time()
    history = []

    random.seed(42)
    # Example Instance
    instance_name = "X-n317-k53.vrp"
    bks = get_bks(instance_name)
    
    # Generate random initial solution and initialize instance
    start = time.time()
    random_routes, instance = generate_random_solution(instance_name)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Random Solution", random_routes, instance, history, runtime=elapsed, bks=bks)

    # Generate Randomized Savings Solution
    """
    start = time.time()
    savings_routes = randomized_savings(instance, alpha=0.3)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Savings Solution", savings_routes, instance, history, runtime=elapsed, bks=bks)
    """
    
    # Standalone Local Search - Swap-based
    """start = time.time()
    ls_swap_routes = ls_with_swaps(instance, copy.deepcopy(savings_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Local Search (Swaps)", ls_swap_routes, instance, history, runtime=elapsed, bks=bks)

    # Standalone Local Search - 2-Opt
    start = time.time()
    ls_2opt_routes = ls_with_2opt(instance, copy.deepcopy(savings_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Local Search (2-Opt)", ls_2opt_routes, instance, history, runtime=elapsed, bks=bks)

    """
    # Local Search Pipeline - Swap -> 2-Opt
    start = time.time()
    ls_pipeline_routes = hybrid_ls(instance, copy.deepcopy(random_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Local Search Pipeline", ls_pipeline_routes, instance, history, runtime=elapsed, bks=bks)
    """

    # Iterated Local Search - Using Hybrid LS by default
    start = time.time()
    ils_routes = iterated_local_search(instance, copy.deepcopy(savings_routes), ls=hybrid_ls, it=50, destroy_factor=0.1)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Iterated LS", ils_routes, instance, history, runtime=elapsed, bks=bks)

    """
    # Standalone Simulated Annealing
    start = time.time()
    sa_routes = simulated_annealing(instance, copy.deepcopy(random_routes))
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Standalone SA", sa_routes, instance, history, runtime=elapsed, bks=bks)

    # LS + Simulated Annealing
    start = time.time()
    ls_sa_routes = simulated_annealing(instance, ls_pipeline_routes)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("LS + SA", ls_sa_routes, instance, history, runtime=elapsed, bks=bks)

    # Simulated Annealing + LS
    start = time.time()
    sa_ls_routes = hybrid_ls(instance, copy.deepcopy(sa_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("SA + LS", sa_ls_routes, instance, history, runtime=elapsed, bks=bks)

    # LS + Simulated Annealing + LS
    start = time.time()
    ls_sa_ls_routes = hybrid_ls(instance, ls_sa_routes, 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("LS + SA + LS", ls_sa_ls_routes, instance, history, runtime=elapsed, bks=bks)

    """
    # Tabu Search
    start = time.time()
    tabu_routes = tabu_search(instance, copy.deepcopy(random_routes), 50)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Standalone Tabu Search", tabu_routes, instance, history, runtime=elapsed, bks=bks)
    
    # Fast LNS
    start = time.time()
    fast_lns_routes = fast_lns(instance, copy.deepcopy(random_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Fast LNS", fast_lns_routes, instance, history, runtime=elapsed, bks=bks)

    # Smart LNS
    start = time.time()
    smart_lns_routes = smart_lns(instance, copy.deepcopy(random_routes), 100)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Smart LNS", smart_lns_routes, instance, history, runtime=elapsed, bks=bks)
    
    # Fast LNS + ILS Hybrid
    start = time.time()
    fast_lns_ils_routes = iterated_local_search(instance, fast_lns_routes)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Fast LNS + ILS", fast_lns_ils_routes, instance, history, runtime=elapsed, bks=bks)

    # Smart LNS + ILS Hybrid
    start = time.time()
    smart_lns_ils_routes = iterated_local_search(instance, smart_lns_routes)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Smart LNS + ILS", smart_lns_ils_routes, instance, history, runtime=elapsed, bks=bks)

    # Genetic Algorithm
    start = time.time()
    routes_ga = genetic_algorithm(instance, pop_size = 40)
    elapsed = round((time.time() - start) / 60, 4)
    log_results("Genetic Algorithm", routes_ga, instance, history, runtime=elapsed, bks=bks)
    """

    total_runtime = round((time.time() - total_start) / 60, 4)

    # Print overview of all results
    print("\n==================== Heuristic Comparison ====================")
    print(f"\nInstance Name: {instance_name}")
    print(f"Best Known Solution: {bks}")
    print(f"\n{'Label':<35} | {'Cost':<10} | {'Gap (%)':<8} | {'Time (min)':<10}")
    print("-" * 62)
    for label, cost, gap, runtime, _ in history:
        gap_str = f"{gap:.2f}" if gap is not None else "N/A"
        print(f"{label:<35} | {cost:<10.2f} | {gap_str:<8}% | {runtime:<10.4f}")

    print(f"\nTotal Runtime in Minutes: {total_runtime}")

    # Identify best solution from history and plot this one
    best_label, best_cost, _, _, best_routes = min(history, key=lambda x: x[1])
    plot_routes(instance, best_routes, title=f"Best: {best_label} (Cost: {best_cost})")

if __name__ == "__main__":
    print("started...")
    main()

