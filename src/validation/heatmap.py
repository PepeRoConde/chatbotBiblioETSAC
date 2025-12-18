import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar el CSV
df = pd.read_csv('memoria/imaxes/grid_search_results.csv')

# Definir las métricas a visualizar
metrics = ['avg_recall', 'avg_precision', 'avg_mrr', 'avg_faithfulness', 'avg_relevance']
metric_labels = ['Recall', 'Precision', 'MRR', 'Faithfulness', 'Relevance']

# Crear una figura con subplots para cada métrica
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    # Crear pivot table para el heatmap
    pivot_table = df.pivot_table(
        values=metric,
        index='chunk_size',
        columns='bm25_weight',
        aggfunc='mean'
    )
    
    # Crear heatmap con colores rosa y gris
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.3f',
        cmap='RdPu',  # Escala de grises a rosas
        cbar_kws={'label': label},
        ax=axes[idx],
        vmin=0,
        vmax=1
    )
    
    axes[idx].set_title(f'{label} por BM25 Weight y Chunk Size', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('BM25 Weight', fontsize=10)
    axes[idx].set_ylabel('Chunk Size', fontsize=10)

# Ocultar el último subplot si no se usa
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('memoria/imaxes/heatmaps/metricas_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Opcional: Crear un heatmap combinado con todas las métricas normalizadas
fig, ax = plt.subplots(figsize=(14, 8))

# Normalizar todas las métricas para compararlas
df_normalized = df.copy()
for metric in metrics:
    df_normalized[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Crear una métrica promedio
df_normalized['avg_all_metrics'] = df_normalized[[f'{m}_norm' for m in metrics]].mean(axis=1)

# Pivot para el heatmap combinado
pivot_combined = df_normalized.pivot_table(
    values='avg_all_metrics',
    index='chunk_size',
    columns='bm25_weight',
    aggfunc='mean'
)

sns.heatmap(
    pivot_combined,
    annot=True,
    fmt='.3f',
    cmap='RdPu',  # Escala de grises a rosas
    cbar_kws={'label': 'Score Promedio Normalizado'},
    ax=ax
)

ax.set_title('Score Promedio de Todas las Métricas\n(Normalizado)', fontsize=14, fontweight='bold')
ax.set_xlabel('BM25 Weight', fontsize=12)
ax.set_ylabel('Chunk Size', fontsize=12)

plt.tight_layout()
plt.savefig('memoria/imaxes/heatmaps/metricas_combinadas_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()