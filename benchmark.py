import random
import time

from heuristics.construction.random import generate_random_solution
from heuristics.construction.random_savings import randomized_savings
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
benchmark_instances = ["X-n502-k39.vrp", 
                       "X-n524-k153.vrp",
                       "X-n561-k42.vrp",
                       "X-n641-k35.vrp",
                       "X-n685-k75.vrp",
                       "X-n749-k75.vrp",
                       "X-n801-k40.vrp",
                       "X-n856-k95.vrp",
                       "X-n916-k207.vrp",
                       "X-n1001-k43.vrp",]

def main():
    instance_name = "X-n106-k14.vrp"
    instance = vrplib.read_instance("instances/" + instance_name)

    bks = get_bks(instance_name)
    name_no_ext = instance_name.lower().replace(".vrp", "")
    k = int(name_no_ext.split("k")[1])
    n = instance["dimension"]
    avg_route_size = n/k

    # ===== EXECUTION =====
    results = {
        "instance_name": instance_name,
        "dimension": n,
        "depot": instance["depot"],
        "avg_route_size": avg_route_size,
        "heuristics": {}
    }

    start_all = time.time()
    n_iter = 2


    # Generate random initial solution and initialize instance
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        print(f"Starting iteration: {iter+1}")
        # Setting the seeds like this ensures that the result are reproducible
        random.seed(iter+1)
        np.random.seed(iter+1)

        start = time.time()
        random_routes, instance = generate_random_solution(instance_name)
        cost_rand = compute_total_cost(random_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_rand)
        gaps.append(round((cost_rand-bks)/bks*100, 2))
        runtimes.append(elapsed)
        print(f"Random iteration {iter+1} done.")

    results["heuristics"]["Random Solution"] = {
        "costs": costs,
        "times": runtimes,
        "gaps": gaps}


    # Iterated Local Search - Using Hybrid LS by default
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ils_routes = iterated_local_search(instance, random_routes, ls=hybrid_ls, it=n, destroy_factor=0.1)
        cost_ils = compute_total_cost(ils_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ils)
        gaps.append(round((cost_ils-bks)/bks*100, 2))
        runtimes.append(elapsed)
        print(f"ILS iteration {iter+1} done.")

    results["heuristics"]["Iterative Local Search"] = {
        "costs": costs,
        "times": runtimes,
        "gaps": gaps}


    # Standalone Simulated Annealing
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        sa_routes = simulated_annealing(instance, random_routes, n)
        cost_sa = compute_total_cost(sa_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_sa)
        gaps.append(round((cost_sa-bks)/bks*100, 2))
        runtimes.append(elapsed)
        print(f"SA iteration {iter+1} done.")

    results["heuristics"]["Simulated Annealing (SA)"] = {
        "costs": costs,
        "times": runtimes,
        "gaps": gaps}

    
    # Hybrid LS + Simulated Annealing + Hybrid LS
    costs, gaps, runtimes = [], [], []
    for iter in range(n_iter):
        start = time.time()
        ls_routes = hybrid_ls(instance, random_routes)
        ls_sa_routes = simulated_annealing(instance, ls_routes, n)
        ls_sa_ls_routes = hybrid_ls(instance, ls_sa_routes)
        cost_ls_sa_ls = compute_total_cost(ls_sa_ls_routes, instance["edge_weight"])
        elapsed = round((time.time() - start) / 60, 4)

        costs.append(cost_ls_sa_ls)
        gaps.append(round((cost_ls_sa_ls-bks)/bks*100, 2))
        runtimes.append(elapsed)
        print(f"LS+SA+LS iteration {iter+1} done.")

    results["heuristics"]["LS + SA + LS"] = {
        "costs": costs,
        "times": runtimes,
        "gaps": gaps}


    # Tabu Search
    costs_tabu, gaps_tabu, runtimes_tabu = [], [], []
    costs_tabu_hls, gaps_tabu_hls, runtimes_tabu_hls = [],[],[]
    for iter in range(n_iter):
        start = time.time()
        tabu_routes = tabu_search(instance, random_routes,n)
        cost_tabu = compute_total_cost(tabu_routes, instance["edge_weight"])
        elapsed1 = round((time.time() - start) / 60, 4)

        costs_tabu.append(cost_tabu)
        gaps_tabu.append(round((cost_tabu-bks)/bks*100, 2))
        runtimes_tabu.append(elapsed1)
        print(f"Tabu iteration {iter+1} done.")

    # Tabu Search + Hybrid LS
        start = time.time()
        tabu_hls_routes = hybrid_ls(instance, tabu_routes, n)
        cost_tabu_hls = compute_total_cost(tabu_hls_routes, instance["edge_weight"])
        elapsed2 = round((time.time() - start) / 60, 4)

        costs_tabu_hls.append(cost_tabu_hls)
        gaps_tabu_hls.append(round((cost_tabu_hls-bks)/bks*100, 2))
        runtimes_tabu_hls.append(elapsed1 + elapsed2)
        print(f"Tabu+LS iteration {iter+1} done.")

    results["heuristics"]["Tabu Search"] = {
        "costs": costs_tabu,
        "times": runtimes_tabu,
        "gaps": gaps_tabu}
    
    results["heuristics"]["Tabu Search + Hybrid Ls"] = {
        "costs": costs_tabu_hls,
        "times": runtimes_tabu_hls,
        "gaps": gaps_tabu_hls}


    # Fast LNS
    costs_lns, gaps_lns, runtimes_lns = [], [], []
    costs_lns_ils, gaps_lns_ils, runtimes_lns_ils = [], [], []
    for iter in range(n_iter):
        start = time.time()
        fast_lns_routes = fast_lns(instance, random_routes, n)
        cost_fast_lns = compute_total_cost(fast_lns_routes, instance["edge_weight"])
        elapsed1 = round((time.time() - start) / 60, 4)

        costs_lns.append(cost_fast_lns)
        gaps_lns.append(round((cost_fast_lns-bks)/bks*100, 2))
        runtimes_lns.append(elapsed1)
        print(f"Fast LNS iteration {iter+1} done.")

    # Fast LNS + ILS
        start = time.time()
        fast_lns_ils_routes = iterated_local_search(instance, fast_lns_routes, it=n)
        cost_fast_lns_ils = compute_total_cost(fast_lns_ils_routes, instance["edge_weight"])
        elapsed2 = round((time.time() - start) / 60, 4)

        costs_lns_ils.append(cost_fast_lns_ils)
        gaps_lns_ils.append(round((cost_fast_lns_ils-bks)/bks*100, 2))
        runtimes_lns_ils.append(elapsed1 + elapsed2)
        print(f"Fast LNS + ILS iteration {iter+1} done.")

    results["heuristics"]["Fast LNS"] = {
        "costs": costs_lns,
        "times": runtimes_lns,
        "gaps": gaps_lns}
    
    results["heuristics"]["Fast LNS + ILS Hybrid"] = {
        "costs": costs_lns_ils,
        "times": runtimes_lns_ils,
        "gaps": gaps_lns_ils}


    # Smart LNS
    costs_lns, gaps_lns, runtimes_lns = [], [], []
    costs_lns_ils, gaps_lns_ils, runtimes_lns_ils = [], [], []
    for iter in range(n_iter):
        start = time.time()
        smart_lns_routes = smart_lns(instance, random_routes, n)
        cost_smart_lns = compute_total_cost(smart_lns_routes, instance["edge_weight"])
        elapsed1 = round((time.time() - start) / 60, 4)

        costs_lns.append(cost_smart_lns)
        gaps_lns.append(round((cost_smart_lns-bks)/bks*100, 2))
        runtimes_lns.append(elapsed1)
        print(f"Smart LNS iteration {iter+1} done.")

    # Smart LNS + ILS
        start = time.time()
        smart_lns_ils_routes = iterated_local_search(instance, smart_lns_routes, it=n)
        cost_smart_lns_ils = compute_total_cost(smart_lns_ils_routes, instance["edge_weight"])
        elapsed2 = round((time.time() - start) / 60, 4)

        costs_lns_ils.append(cost_smart_lns_ils)
        gaps_lns_ils.append(round((cost_smart_lns_ils-bks)/bks*100, 2))
        runtimes_lns_ils.append(elapsed1+elapsed2)
        print(f"Smart LNS + ILS iteration {iter+1} done.")

    results["heuristics"]["Smart LNS"] = {
        "costs": costs_lns,
        "times": runtimes_lns,
        "gaps": gaps_lns}
    
    results["heuristics"]["Smart LNS + ILS Hybrid"] = {
        "costs": costs_lns_ils,
        "times": runtimes_lns_ils,
        "gaps": gaps_lns_ils}

    # Genetic Algorithm
    costs_ga, gaps_ga, runtimes_ga = [], [], []
    costs_ga_hls, gaps_ga_hls, runtimes_ga_hls = [],[],[]
    for iter in range(n_iter):
        start = time.time()
        routes_ga = genetic_algorithm(instance, 40, n*10)
        cost_ga = compute_total_cost(routes_ga, instance["edge_weight"])
        elapsed1 = round((time.time() - start) / 60, 4)

        costs_ga.append(cost_ga)
        gaps_ga.append(round((cost_ga-bks)/bks*100, 2))
        runtimes_ga.append(elapsed1)
        print(f"GA iteration {iter+1} done.")


    # GA + Hybrid LS
        start = time.time()
        routes_ga_hls = hybrid_ls(instance, routes_ga, n)
        cost_ga_hls = compute_total_cost(routes_ga_hls, instance["edge_weight"])
        elapsed2 = round((time.time() - start) / 60, 4)

        costs_ga_hls.append(cost_ga_hls)
        gaps_ga_hls.append(round((cost_ga_hls-bks)/bks*100, 2))
        runtimes_ga_hls.append(elapsed1+elapsed2)
        print(f"GA+LS iteration {iter+1} done.")

    results["heuristics"]["Genetic Algorithm"] = {
        "costs": costs_ga,
        "times": runtimes_ga,
        "gaps": gaps_ga}
    
    results["heuristics"]["GA + Hybrid LS"] = {
        "costs": costs_ga_hls,
        "times": runtimes_ga_hls,
        "gaps": gaps_ga_hls}

    total_runtime = (time.time() - start_all) / 60  # minuti


    # ===== FINAL PRINTING =====
    print("\n==================== Heuristic Comparison ====================")
    print(f"\nInstance Name: {instance_name}")
    print(f"Best Known Solution: {bks}")
    print(f"\n{'Label':<35} | {'Cost':<10} | {'Gap (%)':<8} | {'Time (min)':<10}")
    print("-" * 62)
    for label in results["heuristics"].keys():
        stats = results["heuristics"][label]
        print(f"{label:<35} | "
            f"{np.mean(stats['costs']):<10.2f} | "
            f"{np.mean(stats['gaps']):<8.2f}% | "
            f"{np.mean(stats['times']):<10.4f}")

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