import random
import time
from multiprocessing import Pool, cpu_count
from copy import deepcopy

from heuristics.construction.random import generate_random_solution
from utils.utils import compute_total_cost, get_bks, convert_ndarrays
from heuristics.improvement.ls import hybrid_ls
from heuristics.metaheuristics.instensifying_components.ils import iterated_local_search
from heuristics.metaheuristics.diversifying_components.simulated_annealing import simulated_annealing
from heuristics.metaheuristics.instensifying_components.tabu import tabu_search
from heuristics.metaheuristics.diversifying_components.genetic_algorithm import genetic_algorithm
from heuristics.metaheuristics.diversifying_components.lns import fast_lns, smart_lns
import json
import numpy as np
import vrplib
import os


# ========================= SETUP =========================
benchmark_instances = ["X-n502-k39.vrp", 
                       "X-n524-k153.vrp",
                       "X-n561-k42.vrp",
                       "X-n641-k35.vrp",
                       "X-n685-k75.vrp",
                       "X-n749-k98.vrp",
                       "X-n801-k40.vrp",
                       "X-n856-k95.vrp",
                       "X-n916-k207.vrp",
                       "X-n1001-k43.vrp",]

# ========================= MAIN =========================
def main():
    for i in range(0,1):
        # Initialize
        instance_name = benchmark_instances[8]
        print("\n",instance_name)
        instance = vrplib.read_instance("instances/" + instance_name)
        bks = get_bks(instance_name)
        name_no_ext = instance_name.lower().replace(".vrp", "")
        k = int(name_no_ext.split("k")[1])
        n = instance["dimension"]
        avg_route_size = n/k

        # Prepare Instance's Result Format 
        results = {
            "instance_name": instance_name,
            "dimension": n,
            "depot": instance["depot"],
            "avg_route_size": avg_route_size,
            "total_benchmark_time": None,
            "heuristics": {}
        }

        # SET NUMBER OF ITERATIONS PER INSTANCE HERE:
        #        vv
        n_iter = 10

        seeds = list(range(1, n_iter + 1))
        start_all = time.time()

        # Run n_iter in Parallel via Multiprocessing
        with Pool(processes=min(cpu_count(), n_iter)) as pool:
            args = [(seed, instance_name, bks, n, k) for seed in seeds]
            all_results = pool.starmap(run_iteration, args)

        # Aggregate Results
        aggregated = {}
        for _, iteration_result in all_results:
            for method, (cost, time_) in iteration_result.items():
                if method not in aggregated:
                    aggregated[method] = {"costs": [], "gaps": [], "times": []}
                aggregated[method]["costs"].append(cost)
                aggregated[method]["gaps"].append(round((cost - bks) / bks * 100, 2))
                aggregated[method]["times"].append(time_)

        results["heuristics"] = aggregated
        
        # Track Benchmarking Runtime
        total_runtime = (time.time() - start_all) / 60
        results["total_benchmark_time"] = total_runtime

        # Print Results
        print("\n==================== Heuristic Comparison ====================")
        print(f"\nInstance Name: {instance_name}")
        print(f"Best Known Solution: {bks}")
        print(f"\n{'Label':<16} | {'Mean Cost':<10} | {'Mean Gap (%)':<12} | {'Mean Time (min)':<16}")
        print("-" * 62)
        for label in results["heuristics"].keys():
            stats = results["heuristics"][label]
            print(f"{label:<16} | "
                f"{np.mean(stats['costs']):<10.2f} | "
                f"{np.mean(stats['gaps']):<11.2f}% | "
                f"{np.mean(stats['times']):<16.4f}")

        print(f"\nTotal Runtime in Minutes: {total_runtime:.2f}")

        # Save Results to JSON
        results = convert_ndarrays(results)
        json_filename = os.path.join("output", "results.json")

        # If file exists and contains a list, upload it and add the new result
        if os.path.exists(json_filename):
            with open(json_filename, "r") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        results_list = data
                    else:
                        results_list = [data]
                except Exception:
                    results_list = []
        else:
            results_list = []

        results_list.append(results)

        with open(json_filename, "w") as f:
            json.dump(results_list, f, indent=4)

# ========================= RUN SINGLE ITERATION =========================
def run_iteration(iter_seed, instance_name, bks, n, k):
    # Setting the seeds like this ensures that the result are reproducible
    random.seed(iter_seed)
    np.random.seed(iter_seed)
    print(f"Starting Iteration {iter_seed} ...")

    # Helper Function to Print Progress Updates
    max_label_len = 38  # Max Label Length
    def print_aligned(label):
        dashes = '-' * max(3, (max_label_len - len(label) + 3))
        print(f"{label} {dashes} ✔️   -----")

    results = {}
    
    # Random Initial Solution
    start = time.time()
    random_routes, instance = generate_random_solution(instance_name)
    cost_rand = compute_total_cost(random_routes, instance["edge_weight"])
    elapsed = round((time.time() - start) / 60, 4)
    results["Random"] = (cost_rand, elapsed)
    print_aligned(f"Random Solution Iteration: {iter_seed}")

    # Iterated Local Search
    start = time.time()
    ils_routes = iterated_local_search(deepcopy(instance), deepcopy(random_routes), ls=hybrid_ls, destroy_factor=0.1)
    cost_ils = compute_total_cost(ils_routes, instance["edge_weight"])
    elapsed = round((time.time() - start) / 60, 4)
    results["ILS"] = (cost_ils, elapsed)
    print_aligned(f"ILS Solution Iteration: {iter_seed}")

    # Standalone Simulated Annealing
    start = time.time()
    sa_routes = simulated_annealing(deepcopy(instance), deepcopy(random_routes))
    cost_sa = compute_total_cost(sa_routes, instance["edge_weight"])
    elapsed = round((time.time() - start) / 60, 4)
    results["SA"] = (cost_sa, elapsed)
    print_aligned(f"SA Solution Iteration: {iter_seed}")

    # Hybrid LS + Simulated Annealing + Hybrid LS
    start = time.time()
    ls_routes = hybrid_ls(deepcopy(instance), deepcopy(random_routes))
    ls_sa_routes = simulated_annealing(deepcopy(instance), ls_routes)
    ls_sa_ls_routes = hybrid_ls(deepcopy(instance), ls_sa_routes)
    cost_ls_sa_ls = compute_total_cost(ls_sa_ls_routes, instance["edge_weight"])
    elapsed = round((time.time() - start) / 60, 4)
    results["LS+SA+LS"] = (cost_ls_sa_ls, elapsed)
    print_aligned(f"LS+SA+LS Solution Iteration: {iter_seed}")

    # Tabu Search
    start = time.time()
    tabu_routes = tabu_search(deepcopy(instance), deepcopy(random_routes))
    cost_tabu = compute_total_cost(tabu_routes, instance["edge_weight"])
    elapsed1 = round((time.time() - start) / 60, 4)
    results["Tabu"] = (cost_tabu, elapsed1)
    print_aligned(f"Tabu Solution Iteration: {iter_seed}")

    # Tabu Search + Hybrid LS
    tabu_hls_routes = hybrid_ls(deepcopy(instance), tabu_routes)
    cost_tabu_hls = compute_total_cost(tabu_hls_routes, instance["edge_weight"])
    elapsed2 = round((time.time() - start) / 60, 4)
    results["Tabu+LS"] = (cost_tabu_hls, elapsed2)
    print_aligned(f"Tabu+LS Solution Iteration: {iter_seed}")

    # Fast LNS
    start = time.time()
    fast_lns_routes = fast_lns(deepcopy(instance), deepcopy(random_routes))
    cost_fast_lns = compute_total_cost(fast_lns_routes, instance["edge_weight"])
    elapsed1 = round((time.time() - start) / 60, 4)
    results["Fast LNS"] = (cost_fast_lns, elapsed1)
    print_aligned(f"Fast LNS Solution Iteration: {iter_seed}")

    # Fast LNS + ILS
    fast_lns_ils_routes = iterated_local_search(deepcopy(instance), fast_lns_routes)
    cost_fast_lns_ils = compute_total_cost(fast_lns_ils_routes, instance["edge_weight"])
    elapsed2 = round((time.time() - start) / 60, 4)
    results["Fast LNS + ILS"] = (cost_fast_lns_ils, elapsed2)
    print_aligned(f"Fast LNS + ILS Solution Iteration: {iter_seed}")

    """# Smart LNS
    start = time.time()
    smart_lns_routes = smart_lns(deepcopy(instance), deepcopy(random_routes))
    cost_smart_lns = compute_total_cost(smart_lns_routes, instance["edge_weight"])
    elapsed1 = round((time.time() - start) / 60, 4)
    results["Smart LNS"] = (cost_smart_lns, elapsed1)
    print_aligned(f"Smart LNS Solution Iteration: {iter_seed}")

    # Smart LNS + ILS
    smart_lns_ils_routes = iterated_local_search(deepcopy(instance), smart_lns_routes)
    cost_smart_lns_ils = compute_total_cost(smart_lns_ils_routes, instance["edge_weight"])
    elapsed2 = round((time.time() - start) / 60, 4)
    results["Smart LNS + ILS"] = (cost_smart_lns_ils, elapsed2)
    print_aligned(f"Smart LNS + ILS Solution Iteration: {iter_seed}")"""

    # Genetic Algorithm
    start = time.time()
    routes_ga = genetic_algorithm(deepcopy(instance), 40, n*5)
    cost_ga = compute_total_cost(routes_ga, instance["edge_weight"])
    elapsed1 = round((time.time() - start) / 60, 4)
    results["GA"] = (cost_ga, elapsed1)
    print_aligned(f"GA Solution Iteration: {iter_seed}")

    # GA + Hybrid LS
    routes_ga_hls = hybrid_ls(deepcopy(instance), routes_ga, n)
    cost_ga_hls = compute_total_cost(routes_ga_hls, instance["edge_weight"])
    elapsed2 = round((time.time() - start) / 60, 4)
    results["GA+LS"] = (cost_ga_hls, elapsed2)
    print_aligned(f"GA+LS Solution Iteration: {iter_seed}")

    return (iter_seed, results)




if __name__ == "__main__":
    main()