import csv
import matplotlib.pyplot as plt
from collections import Counter

CSV_PATH = "crawl/bureaucratic_stats.csv"
OUTPUT_PATH = "numero_palabras_clave_documentos.png"

# Read CSV
counts = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        counts.append(int(row['matched_keywords_count']))

counter = Counter(counts)

# Sort levels
levels = sorted(counter.keys())
sizes = [counter[l] for l in levels]
total_docs = sum(sizes)

# Legend labels
legend_labels = []
for lvl in levels:
    count = counter[lvl]
    pct = (count / total_docs) * 100
    if lvl == 0:
        legend_labels.append(
            f'0 → Non burocrático: {count} ({pct:.1f}%)'
        )
    else:
        legend_labels.append(
            f'{lvl} palabras clave: {count} ({pct:.1f}%)'
        )

# Colors
base_pink = (181/255, 60/255, 135/255)  # #b43b86
colors = []

for lvl in levels:
    if lvl == 0:
        colors.append('#B0B0B0')  # gray
    else:
        factor = min(0.15 * lvl, 0.7)
        color = (
            base_pink[0] + (1 - base_pink[0]) * factor,
            base_pink[1] + (1 - base_pink[1]) * factor,
            base_pink[2] + (1 - base_pink[2]) * factor,
        )
        colors.append(color)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

wedges, _ = ax.pie(
    sizes,
    startangle=90,
    colors=colors
)

ax.set_title(
    'Número de palabras chave atopadas nos documentos',
    fontsize=14,
    fontweight='bold',
    pad=20
)

# Legend
ax.legend(
    wedges,
    legend_labels,
    title="Clasificación",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=11,
    title_fontsize=12
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Gráfico gardado en '{OUTPUT_PATH}'")

