import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns

# --- Set Up ---
with open("output/results.json", "r") as f:
    raw_data = json.load(f)

# We will need that later to 
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

if isinstance(raw_data, dict) and "heuristics" in raw_data:
    data = [raw_data]
else:
    data = raw_data
    # Uncomment below to only get the data from the results.json, which are for our selected benchmarks.
    # data = [entry for entry in raw_data if entry.get("instance_name") in benchmark_instances]


# why exclude?
heuristics_names = [h for h in data[0]["heuristics"].keys() if h.lower() in ["sa", "tabu", "fast lns", "ga"]]  # Exclude 'Random' and 'Smart LNS' heuristics

# --- Compute global min and max for y-axis (GAP), excluding 'Random' and 'Smart LNS' heuristics ---
all_gaps = []
for entry in data:
    heuristics = entry.get("heuristics", {})
    for h_name, h in heuristics.items():
        if h_name.lower() in ["sa","tabu", "fast lns", "ga"]:
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

    # Plot with sorted values, reduce boxplot distance and figure width
    fig, ax = plt.subplots(figsize=(max(6, len(sorted_labels)*0.8), 6))
    box = ax.boxplot(sorted_costs, patch_artist=True, tick_labels=['']*len(sorted_labels), showmeans=False, widths=0.7)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    # Make median line thicker and black
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    ax.set_ylabel("GAP")
    ax.set_xlabel("Number of Nodes")
    ax.set_title(f"GAP distribution for {heuristic}")
    # Remove x-tick labels, keep ticks
    ax.set_xticks(range(1, len(sorted_labels)+1))
    ax.set_xticklabels(['']*len(sorted_labels))
    # Add colorbar legend for number of nodes
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min(sorted_dims), vmax=max(sorted_dims)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Number of Nodes')
    # Add a small margin above the global max to ensure all boxplots are fully visible
    y_margin = (global_max_gap - global_min_gap) * 0.01 if global_max_gap > global_min_gap else 1
    ax.set_ylim(global_min_gap - y_margin, global_max_gap + y_margin)
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

    deposit_keys = sorted(deposit_to_gaps.keys(), key=str)
    data_for_plot = [deposit_to_gaps[k] for k in deposit_keys]
    labels = [str(k) for k in deposit_keys]

    # Use a red scale for string deposits
    import matplotlib
    color_palette = matplotlib.colormaps.get_cmap('Reds')
    if len(deposit_keys) > 1:
        color_indices = np.linspace(0.2, 0.9, len(deposit_keys))  # avoid too light/dark extremes
    else:
        color_indices = [0.6]
    box_colors = [color_palette(idx) for idx in color_indices]

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
    y_margin = (global_max_gap - global_min_gap) * 0.01 if global_max_gap > global_min_gap else 1
    plt.ylim(global_min_gap-y_margin, global_max_gap + y_margin)
    plt.tight_layout()
    plt.show()

# --- Boxplot of gaps for each heuristic, grouped by avg_route_size intervals ---
for heuristic in heuristics_names:
    # Prepare gap lists for each avg_route_size interval
    gaps_0_8 = []
    gaps_8_16 = []
    gaps_16_inf = []
    for entry in data:
        avg_route_size = entry.get("avg_route_size", None)
        heuristics = entry.get("heuristics", {})
        gaps = heuristics.get(heuristic, {}).get("gaps", [])
        if avg_route_size is not None:
            if avg_route_size <= 8:
                gaps_0_8.extend(gaps)
            elif avg_route_size <= 16:
                gaps_8_16.extend(gaps)
            else:
                gaps_16_inf.extend(gaps)

    data_for_plot = [gaps_0_8, gaps_8_16, gaps_16_inf]
    labels = ["[0,7]", "[8, 16]", "[17, 23]"]
    # Use a color gradient for the three intervals
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    box_colors = [cmap(norm(i)) for i in range(3)]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_for_plot, patch_artist=True, tick_labels=labels, showmeans=False)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    # Make median line thicker and black
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    plt.ylabel("GAP")
    plt.xlabel("avg_route_size interval")
    plt.title(f"GAP distribution for {heuristic} by avg_route_size interval")
    plt.xticks(rotation=45, ha='right')
    # Add a small margin above the global max to ensure all boxplots are fully visible
    y_margin = (global_max_gap - global_min_gap) * 0.01 if global_max_gap > global_min_gap else 1
    plt.ylim(global_min_gap-y_margin, global_max_gap + y_margin)
    plt.tight_layout()
    plt.show()

# --- Boxplot of times for each heuristic, grouped by dimension (as before) ---
all_times_global = []
for entry in data:
    heuristics = entry.get("heuristics", {})
    for h_name, h in heuristics.items():
        if h_name.lower() in ["sa","tabu", "fast lns", "ga"]:
            all_times_global.extend(h.get("times", []))
if all_times_global:
    global_min_time = min(all_times_global)
    global_max_time = max(all_times_global)
else:
    global_min_time = 0
    global_max_time = 1
for heuristic in heuristics_names:
    times_per_instance = []
    instance_labels = []
    dimensions = []
    for entry in data:
        dim = entry.get("dimension", "?")
        instance_labels.append(dim)
        dimensions.append(dim)
        heuristics = entry.get("heuristics", {})
        times = heuristics.get(heuristic, {}).get("times", [])
        times_per_instance.append(times)

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
    combined = sorted(zip(dimensions, times_per_instance, instance_labels), key=lambda x: x[0])
    sorted_dims, sorted_times, sorted_labels = zip(*combined)

    # Normalize again for color mapping
    norm = mpl.colors.Normalize(vmin=min(sorted_dims), vmax=max(sorted_dims))
    box_colors = [cmap(norm(d)) for d in sorted_dims]

    fig, ax = plt.subplots(figsize=(max(6, len(sorted_labels)*0.8), 6))
    box = ax.boxplot(sorted_times, patch_artist=True, tick_labels=sorted_labels, showmeans=False, widths=0.7)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of Nodes")
    #ax.set_title(f"Time distribution for {heuristic}")
    ax.set_xticks(range(1, len(sorted_labels)+1))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
    y_margin = (global_max_time - global_min_time) * 0.01 if global_max_time > global_min_time else 1
    ax.set_ylim(global_min_time - y_margin, global_max_time + y_margin)
    plt.tight_layout()
    plt.show()

# --- Boxplot of times for each heuristic, grouped by deposit value ---
for heuristic in heuristics_names:
    deposit_to_times = {}
    for entry in data:
        deposits = entry.get("depot", [])
        heuristics = entry.get("heuristics", {})
        times = heuristics.get(heuristic, {}).get("times", [])
        for dep in deposits:
            if dep not in deposit_to_times:
                deposit_to_times[dep] = []
            deposit_to_times[dep].extend(times)

    deposit_keys = sorted(deposit_to_times.keys(), key=str)
    data_for_plot = [deposit_to_times[k] for k in deposit_keys]
    labels = [str(k) for k in deposit_keys]

    # Use a red scale for string deposits
    import matplotlib
    color_palette = matplotlib.colormaps.get_cmap('Reds')
    if len(deposit_keys) > 1:
        color_indices = np.linspace(0.3, 0.85, len(deposit_keys))
    else:
        color_indices = [0.6]
    box_colors = [color_palette(idx) for idx in color_indices]

    plt.figure(figsize=(max(8, len(labels)*1.2), 6))
    box = plt.boxplot(data_for_plot, patch_artist=True, tick_labels=labels, showmeans=False)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    plt.ylabel("Time (s)")
    plt.xlabel("Deposit")
    plt.title(f"Time distribution for {heuristic} by deposit")
    plt.xticks(rotation=45, ha='right')
    y_margin = (global_max_time - global_min_time) * 0.01 if global_max_time > global_min_time else 1
    plt.ylim(global_min_time-y_margin, global_max_time + y_margin)
    plt.tight_layout()
    plt.show()

# --- Boxplot of times for each heuristic, grouped by avg_route_size intervals ---
for heuristic in heuristics_names:
    times_0_8 = []
    times_8_16 = []
    times_16_inf = []
    for entry in data:
        avg_route_size = entry.get("avg_route_size", None)
        heuristics = entry.get("heuristics", {})
        times = heuristics.get(heuristic, {}).get("times", [])
        if avg_route_size is not None:
            if avg_route_size <= 8:
                times_0_8.extend(times)
            elif avg_route_size <= 16:
                times_8_16.extend(times)
            else:
                times_16_inf.extend(times)

    data_for_plot = [times_0_8, times_8_16, times_16_inf]
    labels = ["[0,7]", "[8, 16]", "[17, 23]"]
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    box_colors = [cmap(norm(i)) for i in range(3)]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(data_for_plot, patch_artist=True, tick_labels=labels, showmeans=False)
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    plt.ylabel("Time (s)")
    plt.xlabel("avg_route_size interval")
    #plt.title(f"Time distribution for {heuristic} by avg_route_size interval")
    plt.xticks(rotation=45, ha='right')
    y_margin = (global_max_time - global_min_time) * 0.01 if global_max_time > global_min_time else 1
    plt.ylim(global_min_time-y_margin, global_max_time + y_margin)
    plt.tight_layout()
    plt.show()

# --- Create summary table: instances x metaheuristics, with mean GAP and mean TIME ---
# Select metaheuristics (heuristics_names)
metaheuristics = heuristics_names

# Prepare table data
rows = []
instance_names = [entry.get("instance_name", f"instance_{i}") for i, entry in enumerate(data)]

for idx, entry in enumerate(data):
    row = {}
    for heuristic in metaheuristics:
        h = entry.get("heuristics", {}).get(heuristic, {})
        gaps = h.get("gaps", [])
        times = h.get("times", [])
        mean_gap = round(np.mean(gaps), 2) if gaps else None
        mean_time = round(np.mean(times), 2) if times else None
        row[(heuristic, "GAP")] = mean_gap
        row[(heuristic, "TIME")] = mean_time
    rows.append(row)

# Build MultiIndex columns
columns = pd.MultiIndex.from_tuples(row.keys())
df = pd.DataFrame(rows, index=instance_names, columns=columns)

print("\nSummary Table:")
print(df)

# Export the summary table to Excel
excel_path = 'summary_table.xlsx'
df.to_excel(excel_path)
print(f"Table exported on: {excel_path}")

# --- Plot the summary table as a heatmap ---
# Plot GAP values
plt.figure(figsize=(2+len(metaheuristics)*2, 1+len(instance_names)*0.5))
gap_df = df.xs('GAP', axis=1, level=1)
sns.heatmap(gap_df, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Mean GAP'})
plt.title('Mean GAP per Instance and Metaheuristic')
plt.ylabel('Instance')
plt.xlabel('Metaheuristic')
plt.tight_layout()
plt.show()

# Plot TIME values
plt.figure(figsize=(2+len(metaheuristics)*2, 1+len(instance_names)*0.5))
time_df = df.xs('TIME', axis=1, level=1)
sns.heatmap(time_df, annot=True, fmt='.2f', cmap='OrRd', cbar_kws={'label': 'Mean TIME (s)'})
plt.title('Mean TIME per Instance and Metaheuristic')
plt.ylabel('Instance')
plt.xlabel('Metaheuristic')
plt.tight_layout()
plt.show()