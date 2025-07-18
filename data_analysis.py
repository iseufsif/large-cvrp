import json
import pandas as pd
import matplotlib.pyplot as plt

# --- Set Up ---
with open("output/results.json", "r") as f:
    raw_data = json.load(f)

if isinstance(raw_data, dict) and "heuristics" in raw_data:
    data = [raw_data]
else:
    data = raw_data

heuristics_names = list(data[0]["heuristics"].keys())

# --- Gaps Boxplot for each instance, for each heuristics---
for heuristic in heuristics_names:
    # Raccogli i costi per ogni instance
    costs_per_instance = []
    instance_labels = []
    for entry in data:
        instance_labels.append(entry.get("dimension", "?"))
        heuristics = entry.get("heuristics", {})
        costs = heuristics.get(heuristic, {}).get("gaps", [])
        costs_per_instance.append(costs)
    plt.figure(figsize=(max(8, len(instance_labels)*1.2), 6))
    plt.boxplot(costs_per_instance, tick_labels=instance_labels, showmeans=True)
    plt.ylabel("GAP")
    plt.xlabel("Number of Nodes")
    plt.title(f"GAP distribution for {heuristic}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

