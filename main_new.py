import random
import time

from heuristics.construction.random import generate_random_solution
from heuristics.construction.savings import randomized_savings
from utils.utils import compute_total_cost, write_solution, print_solution, log_results, get_bks, convert_ndarrays
from utils.plot import plot_routes
from heuristics.metaheuristics.instensifying_components.ls import ls_with_2opt, ls_with_swaps, hybrid_ls
from heuristics.metaheuristics.instensifying_components.ils import iterated_local_search
from heuristics.metaheuristics.diversifying_components.simulated_annealing import simulated_annealing
from heuristics.metaheuristics.instensifying_components.tabu import tabu_search
from heuristics.metaheuristics.diversifying_components.genetic_algorithm import genetic_algorithm
from heuristics.metaheuristics.diversifying_components.hybrid_genetic_search import HGS
from heuristics.metaheuristics.diversifying_components.lns import fast_lns, smart_lns
from utils.tsp_solvers_for_GA import tsp_solver_nn
import json
import numpy as np
import vrplib
import os
import pandas as pd

# Supponi che queste funzioni siano già definite
# from solvers import greedy_vrp_solver, hgs_solver, ...
# from utils import compute_total_cost, is_feasible

# ===== SETUP =====
def main():
    instance_name = "A-n33-k5.vrp"
    instance = vrplib.read_instance("instances/" + instance_name)

    bks = get_bks(instance_name)
    name_no_ext = instance_name.lower().replace(".vrp", "")
    k = int(name_no_ext.split("k")[1])
    avg_route_size = instance["dimension"] / k

    # ===== EXECUTION =====
    results = {
        "instance_name": instance_name,
        "dimension": instance["dimension"],
        "depot": instance["depot"],
        "avg_route_size": avg_route_size,
        "heuristics": {}
    }

    start_all = time.time()

    n_iter = 2
    # Generate random initial solution and initialize instance
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        random_routes, instance = generate_random_solution(instance_name)
        cost_rand = compute_total_cost(random_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_rand)
        gaps.append(round((cost_rand-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Random Solution"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Generate Randomized Savings Solution
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        sav_routes = randomized_savings(instance, 0.3)
        cost_sav = compute_total_cost(sav_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_sav)
        gaps.append(round((cost_sav-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Randomized Savings Solution"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}
    
    # Standalone Local Search - Swap-based
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ls_swap_routes = ls_with_swaps(instance, sav_routes, 100)
        cost_ls_swap = compute_total_cost(ls_swap_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ls_swap)
        gaps.append(round((cost_ls_swap-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Local Search with Swaps"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Standalone Local Search - 2-Opt
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ls_2opt_routes = ls_with_2opt(instance, sav_routes, 100)
        cost_ls_2opt = compute_total_cost(ls_2opt_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ls_2opt)
        gaps.append(round((cost_ls_2opt-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Local Search with Two-Opt"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Local Search Pipeline - Swap -> 2-Opt
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ls_pipeline_routes = hybrid_ls(instance, sav_routes, 100)
        cost_hybrid_ls = compute_total_cost(ls_pipeline_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_hybrid_ls)
        gaps.append(round((cost_hybrid_ls-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Local Search Pipeline"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Iterated Local Search - Using Hybrid LS by default
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        print(iter)
        start = time.time()
        ils_routes = iterated_local_search(instance, sav_routes, ls=hybrid_ls, it=50, destroy_factor=0.1)
        cost_ils = compute_total_cost(ils_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ils)
        gaps.append(round((cost_ils-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Iterative Local Search"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Standalone Simulated Annealing
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        sa_routes = simulated_annealing(instance, sav_routes)
        cost_sa = compute_total_cost(sa_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_sa)
        gaps.append(round((cost_sa-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Simulated Annealing"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # LS + Simulated Annealing
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ls_sa_routes = simulated_annealing(instance, ls_pipeline_routes)
        cost_ls_sa = compute_total_cost(ls_sa_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ls_sa)
        gaps.append(round((cost_ls_sa-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["LS + Simulated Annealing"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Tabu Search
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        tabu_routes = tabu_search(instance, sav_routes, 50)
        cost_tabu = compute_total_cost(tabu_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_tabu)
        gaps.append(round((cost_tabu-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Tabu Search"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}


    # Fast LNS
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        fast_lns_routes = fast_lns(instance, sav_routes, 100)
        cost_fast_lns = compute_total_cost(fast_lns_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_fast_lns)
        gaps.append(round((cost_fast_lns-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Fast LNS"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Smart LNS
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        smart_lns_routes = smart_lns(instance, sav_routes, 100)
        cost_smart_lns = compute_total_cost(smart_lns_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_smart_lns)
        gaps.append(round((cost_smart_lns-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Smart LNS"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Fast LNS + ILS Hybrid
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        fast_lns_ils_routes = iterated_local_search(instance, fast_lns_routes)
        cost_fast_lns_ils = compute_total_cost(fast_lns_ils_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_fast_lns_ils)
        gaps.append(round((cost_fast_lns_ils-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Fast LNS + ILS Hybrid"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Smart LNS + ILS Hybrid
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        smart_lns_ils_routes = iterated_local_search(instance, smart_lns_routes)
        cost_smart_lns_ils = compute_total_cost(smart_lns_ils_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_smart_lns_ils)
        gaps.append(round((cost_smart_lns_ils-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Smart LNS + ILS Hybrid"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    # Genetic Algorithm
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        routes_ga = genetic_algorithm(instance, pop_size = 40)
        cost_ga = compute_total_cost(routes_ga, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ga)
        gaps.append(round((cost_ga-bks)/bks*100, 2))
        runtimes.append(elapsed)

    results["heuristics"]["Genetic Algorithm"] = {
        "avg_cost": round(np.mean(costs), 4),
        "std_cost": round(np.std(costs), 4),
        "avg_time": round(np.mean(runtimes), 4),
        "std_time": round(np.std(runtimes), 4),
        "avg_gap": round(np.mean(gaps), 4),
        "std_gap": round(np.std(gaps),4)}

    total_runtime = (time.time() - start_all) / 60  # minuti

    # ===== FINAL PRINTING =====
    print("\n==================== Heuristic Comparison ====================")
    print(f"\nInstance Name: {instance_name}")
    print(f"Best Known Solution: {bks}")
    print(f"\n{'Label':<25} | {'Cost':<10} | {'Gap (%)':<8} | {'Time (min)':<10}")
    print("-" * 62)
    for label in results["heuristics"].keys():
        stats = results["heuristics"][label]
        print(f"{label:<25} | "
            f"{stats['avg_cost']:<10.2f} | "
            f"{stats['avg_gap']:<8.2f}% | "
            f"{stats['avg_time']:<10.4f}")

    print(f"\nTotal Runtime in Minutes: {total_runtime:.2f}")

    # ===== JSON SAVING =====
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

if __name__ == "__main__":
    main()