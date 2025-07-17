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

gap_data = {h: [] for h in heuristics_names}
for entry in data:
    for h in heuristics_names:
        gap_data[h].append(entry["heuristics"][h]["avg_gap"])

# --- Box Plot of average gap per heuristic ---
labels = list(gap_data.keys())
data_for_plot = [gap_data[h] for h in labels]

plt.figure(figsize=(max(8, len(labels)*1.5), 6))
plt.boxplot(data_for_plot, tick_labels=labels, showmeans=True)
plt.ylabel('GAP (%)')

plt.title('GAP Distribution (%) per Heuristics')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Bar Plot: average time among all heuristics per instance ---
instance_names = []
avg_times = []
for entry in data:
    instance_names.append(entry.get("instance_name", "?"))
    heuristics = entry.get("heuristics", {})
    times = []
    for h in heuristics.values():
        t = h.get("avg_time")
        if t is not None:
            times.append(t)
    avg_times.append(sum(times)/len(times) if times else 0)

plt.figure(figsize=(max(8, len(instance_names)*1.2), 6))
plt.bar(instance_names, avg_times, color='skyblue')
plt.ylabel("Average Time (s)")
plt.xlabel("Instance")
plt.title("Average time across all heuristics per instance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Boxplot average time Genetic Algorithm per avg_route_size > 10 and <= 10 ---
ga_times_route_gt10 = []
ga_times_route_le10 = []
for entry in data:
    avg_route_size = entry.get("avg_route_size", None)
    ga_time = entry["heuristics"].get("Genetic Algorithm", {}).get("avg_time")
    if ga_time is not None and avg_route_size is not None:
        if avg_route_size > 10:
            ga_times_route_gt10.append(ga_time)
        else:
            ga_times_route_le10.append(ga_time)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
axes[0].boxplot(ga_times_route_le10, showmeans=True)
axes[0].set_title("Genetic Algorithm - avg_route_size ≤ 10")
axes[0].set_ylabel("Time (s)")
axes[0].set_xticks([1])
axes[0].set_xticklabels(["avg_route_size ≤ 10"])

axes[1].boxplot(ga_times_route_gt10, showmeans=True)
axes[1].set_title("Genetic Algorithm - avg_route_size > 10")
axes[1].set_xticks([1])
axes[1].set_xticklabels(["avg_route_size > 10"])

fig.suptitle("Time distribution for Genetic Algorithm by avg_route_size")
plt.tight_layout()
plt.show()


# --- Boxplot average time Genetic Algorithm per depot type ---
from collections import defaultdict
depot_times = defaultdict(list)
for entry in data:
    depot_list = entry.get("depot", [])
    ga_time = entry["heuristics"].get("Genetic Algorithm", {}).get("avg_time")
    if ga_time is not None:
        for depot_val in depot_list:
            depot_times[depot_val].append(ga_time)

depot_keys = sorted(depot_times.keys())
data_for_plot = [depot_times[k] for k in depot_keys]
labels = [f"depot={k}" for k in depot_keys]

plt.figure(figsize=(max(8, len(labels)*2), 6))
plt.boxplot(data_for_plot, tick_labels=labels, showmeans=True)
plt.ylabel("Time (s)")
plt.title("Time distribution for Genetic Algorithm per depot")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
