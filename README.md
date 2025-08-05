# Large-Scale CVRP Solver with Metaheuristics

📦 **Course**: Computational Logistics (@TUM)
🐍 **Python**: 3.13  
📁 **Author**: Alberto Adro, Lucca Altendeitering

---

# 🚚 Large-Scale CVRP Solver using Metaheuristics

This repository implements a modular and scalable solver for large-scale **Capacitated Vehicle Routing Problems (CVRP)**. It leverages various metaheuristic approaches, including hybrid strategies, to solve complex routing instances efficiently. This project was developed as part of the final project in the course **Computational Logistics**.

Implemented algorithms include:
- Iterated Local Search (ILS)
- Simulated Annealing (SA)
- Tabu Search
- Genetic Algorithm (GA)
- Fast and Smart Large Neighborhood Search (LNS)
- Hybrid Metaheuristic Pipelines (e.g., GA + LS, Fast LNS + ILS, LS + SA + LS)

You can evaluate algorithms on individual instances, benchmark multiple methods in parallel across large datasets, and visualize the results interactively with statistical breakdowns.

---

## 🛠️ Setup and Installation

Start by cloning the repository and setting up your environment:

```bash
git clone https://github.com/iseufsif/large-cvrp.git
cd large-cvrp
```

Create a virtual environment and activate it:

```bash
python3.13 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

This will install all dependencies for running heuristics, benchmarking, and plotting results. Make sure you are using **Python 3.13** or adjust your environment accordingly.

---

## ▶️ Run Single Instance (`main.py`)

Use `main.py` to run and test individual CVRP instances with your desired metaheuristics. By default, the script:

- Loads an instance (e.g., `X-n317-k53.vrp`)
- Generates a random or savings-based initial solution
- Applies various heuristics such as Hybrid Local Search, ILS, SA, Tabu, GA, LNS
- Logs results with total cost, gap from BKS, and runtime
- Plots the best solution using matplotlib

You can enable/disable algorithms directly in the script by commenting/uncommenting blocks. The output includes a summary table printed to the terminal and a visualized route for the best solution.

This file can also be used to conduct basic parameter finetuning for each heuristic.

Run it with:

```bash
python main.py
```

You may customize the instance name, number of iterations, and methods within the file. All results are tracked in a local `history` list, and the best solution is plotted automatically.

---

## 📊 Run Benchmark (`benchmark.py`)

The `benchmark.py` script runs large-scale performance evaluations across multiple instances in parallel. It supports 10 repetitions per instance and evaluates a wide range of heuristics and hybrid methods using multiprocessing for efficiency.

The instances we ran for our project benchmark are as follows:

- "X-n502-k39.vrp", 
- "X-n524-k153.vrp",
- "X-n561-k42.vrp",
- "X-n641-k35.vrp",
- "X-n685-k75.vrp",
- "X-n716-k35.vrp",
- "X-n749-k98.vrp",
- "X-n801-k40.vrp",
- "X-n856-k95.vrp",
- "X-n916-k207.vrp"

For each repetition, the following steps are performed:
- Generate a random initial solution
- Apply ILS, SA, Tabu, Fast LNS, GA, and hybrid variants like LS+SA+LS, GA+LS, Tabu+LS
- Measure cost, compute GAP to BKS, and record runtime

All results are stored in `output/results.json`. If the file already exists, new results are appended. A summary of mean cost, mean GAP, and runtime per heuristic is printed after each instance.

To run the benchmark:

```bash
python benchmark.py
```

You can edit the number of repetitions (`n_iter`) and the selected heuristics inside the script. This determines the number of runs for each instance to account for randomness in our evaluation. The benchmark is designed to be modular and easily extended with new algorithm variants.

---

## 📈 Visualize Results (`visuals.py`)

Once benchmarking is complete, use `visuals.py` to generate plots and tables that summarize performance across all methods and instances.

The script loads `output/results.json` and produces:

- Boxplots of GAP grouped by instance size (number of nodes)
- GAP grouped by depot ID and by average route size interval
- Runtime distributions for each heuristic
- Summary tables (instance × heuristic) showing average GAP and runtime
- Export of the summary table to `summary_table.xlsx`
- Heatmaps of mean GAP and mean time

Run the script with:

```bash
python visuals.py
```

Make sure `output/results.json` exists and contains valid results from `benchmark.py`. The script uses Seaborn and Matplotlib for styling and creates publication-ready plots for all statistical evaluations.

The visuals provide insight into:
- Which heuristics perform best on which instance sizes
- How depot or route structure influences performance
- Runtime vs. solution quality trade-offs

---

This project provides a complete experimentation pipeline for solving and analyzing large-scale CVRP instances using advanced metaheuristics. You are encouraged to customize the pipeline for additional heuristics, parameter tuning, or new benchmarking strategies.




