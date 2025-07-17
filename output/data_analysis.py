import json
import pandas as pd
import matplotlib.pyplot as plt

with open("output/results.json", "r") as f:
    raw_data = json.load(f)

# Se è un dizionario singolo, lo mettiamo in una lista
if isinstance(raw_data, dict) and "heuristics" in raw_data:
    data = [raw_data]
else:
    data = raw_data

# Estrai i nomi delle euristiche
heuristics_names = list(data[0]["heuristics"].keys())

# Costruisci una lista di liste: ogni lista contiene i gap di una euristica su tutte le istanze
gap_data = {h: [] for h in heuristics_names}
for entry in data:
    for h in heuristics_names:
        gap_data[h].append(entry["heuristics"][h]["avg_gap"])

# Prepara i dati per il boxplot
labels = list(gap_data.keys())
data_for_plot = [gap_data[h] for h in labels]

plt.figure(figsize=(max(8, len(labels)*1.5), 6))
plt.boxplot(data_for_plot, tick_labels=labels, showmeans=True)
plt.ylabel('GAP (%)')
plt.xlabel('Euristica')
plt.title('Distribuzione GAP (%) per Euristica')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
