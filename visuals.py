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

heuristics_names = [h for h in data[0]["heuristics"].keys() if h.lower() not in ["random", "smart lns"]]  # Exclude 'Random' and 'Smart LNS' heuristics

# --- Gaps Boxplot for each instance, for each heuristic ---
import numpy as np
import matplotlib as mpl



# --- Compute global min and max for y-axis (GAP), excluding 'Random' and 'Smart LNS' heuristics ---
all_gaps = []
for entry in data:
    heuristics = entry.get("heuristics", {})
    for h_name, h in heuristics.items():
        if h_name.lower() not in ["random", "smart lns"]:
            all_gaps.extend(h.get("gaps", []))
if all_gaps:
    global_min_gap = min(all_gaps)
    global_max_gap = max(all_gaps)
else:
    global_min_gap = 0
    global_max_gap = 1

# --- Boxplot of gaps for each heuristic, grouped by dimension (as before) ---
for heuristic in heuristics_names:
    costs_per_instance = []
    instance_labels = []
    dimensions = []
    for entry in data:
        dim = entry.get("dimension", "?")
        instance_labels.append(dim)
        dimensions.append(dim)
        heuristics = entry.get("heuristics", {})
        costs = heuristics.get(heuristic, {}).get("gaps", [])
        costs_per_instance.append(costs)

    # Normalize dimensions for color map
    dims = np.array([d for d in dimensions if isinstance(d, (int, float))])
    if len(dims) > 0:
        min_dim, max_dim = dims.min(), dims.max()
    else:
        min_dim, max_dim = 0, 1
    norm = mpl.colors.Normalize(vmin=min_dim, vmax=max_dim)
    cmap = mpl.cm.Greens
    box_colors = []
    for d in dimensions:
        if isinstance(d, (int, float)):
            box_colors.append(cmap(norm(d)))
        else:
            box_colors.append((1,1,1,1))  # white if not numerical

    # Sort by dimension
    combined = sorted(zip(dimensions, costs_per_instance, instance_labels), key=lambda x: x[0])
    sorted_dims, sorted_costs, sorted_labels = zip(*combined)

    # Normalize again for color mapping
    norm = mpl.colors.Normalize(vmin=min(sorted_dims), vmax=max(sorted_dims))
    box_colors = [cmap(norm(d)) for d in sorted_dims]

    # Plot with sorted values
    plt.figure(figsize=(max(8, len(sorted_labels)*1.2), 6))
    box = plt.boxplot(sorted_costs, patch_artist=True, tick_labels=sorted_labels, showmeans=False)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    # Make median line thicker and black
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    plt.ylabel("GAP")
    plt.xlabel("Number of Nodes")
    plt.title(f"GAP distribution for {heuristic}")
    plt.xticks(rotation=45, ha='right')
    # Add a small margin above the global max to ensure all boxplots are fully visible
    y_margin = (global_max_gap - global_min_gap) * 0.01 if global_max_gap > global_min_gap else 1
    plt.ylim(global_min_gap - y_margin, global_max_gap + y_margin)
    plt.tight_layout()
    plt.show()

# --- Boxplot of gaps for each heuristic, grouped by deposit value ---
for heuristic in heuristics_names:
    deposit_to_gaps = {}

    for entry in data:
        deposits = entry.get("depot", [])
        heuristics = entry.get("heuristics", {})
        gaps = heuristics.get(heuristic, {}).get("gaps", [])
        for dep in deposits:
            if dep not in deposit_to_gaps:
                deposit_to_gaps[dep] = []
            deposit_to_gaps[dep].extend(gaps)

    deposit_keys = sorted(deposit_to_gaps.keys())
    data_for_plot = [deposit_to_gaps[k] for k in deposit_keys]
    labels = [f"deposit={k}" for k in deposit_keys]

    # Colors from light yellow to dark red based on deposit value
    norm = mpl.colors.Normalize(vmin=min(deposit_keys), vmax=max(deposit_keys))
    cmap = mpl.cm.OrRd
    box_colors = [cmap(norm(k)) for k in deposit_keys]

    plt.figure(figsize=(max(8, len(labels)*1.2), 6))
    box = plt.boxplot(data_for_plot, patch_artist=True, tick_labels=labels, showmeans=False)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    # Make median line thicker and black
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    plt.ylabel("GAP")
    plt.xlabel("Deposit")
    plt.title(f"GAP distribution for {heuristic} by deposit")
    plt.xticks(rotation=45, ha='right')
    # Add a small margin above the global max to ensure all boxplots are fully visible
    y_margin = (global_max_gap - global_min_gap) * 0.005 if global_max_gap > global_min_gap else 1
    plt.ylim(global_min_gap, global_max_gap + y_margin)
    plt.tight_layout()
    plt.show()
