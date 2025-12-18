import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Cargar el CSV
df = pd.read_csv('memoria/imaxes/grid_search_results.csv')

# Crear figura con dos subplots 3D
fig = plt.figure(figsize=(16, 7))

metrics = ['avg_faithfulness', 'avg_relevance']
titles = ['Faithfulness', 'Relevance']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
    
    # Preparar datos
    x = df['bm25_weight'].values
    y = df['chunk_size'].values
    z = df[metric].values
    
    # Crear malla para interpolación suave
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolar valores Z
    Z = griddata((x, y), z, (X, Y), method='cubic')
    
    # Crear superficie
    surf = ax.plot_surface(X, Y, Z, cmap='RdPu', 
                           edgecolor='none', alpha=0.9,
                           vmin=0, vmax=1)
    
    # Añadir puntos de datos reales
    ax.scatter(x, y, z, c='darkred', s=50, alpha=0.6, edgecolors='black')
    
    # Configurar etiquetas y título
    ax.set_xlabel('BM25 Weight', fontsize=10, labelpad=10)
    ax.set_ylabel('Chunk Size', fontsize=10, labelpad=10)
    ax.set_zlabel(title, fontsize=10, labelpad=10)
    ax.set_title(f'{title} por BM25 Weight y Chunk Size', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # Añadir barra de color
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Ajustar el ángulo de vista
    ax.view_init(elev=25, azim=45)
    
    # Configurar límites del eje Z
    ax.set_zlim(0, 1)

plt.tight_layout()
plt.savefig('memoria/imaxes/heatmaps/surface_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Opcional: Crear versión con contornos proyectados
fig = plt.figure(figsize=(16, 7))

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
    
    x = df['bm25_weight'].values
    y = df['chunk_size'].values
    z = df[metric].values
    
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')
    
    # Superficie con wireframe
    surf = ax.plot_surface(X, Y, Z, cmap='RdPu', 
                           alpha=0.7, vmin=0, vmax=1)
    
    # Añadir contornos en la base
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='RdPu', alpha=0.5)
    
    # Puntos de datos
    ax.scatter(x, y, z, c='darkred', s=50, alpha=0.8, edgecolors='black')
    
    ax.set_xlabel('BM25 Weight', fontsize=10, labelpad=10)
    ax.set_ylabel('Chunk Size', fontsize=10, labelpad=10)
    ax.set_zlabel(title, fontsize=10, labelpad=10)
    ax.set_title(f'{title} - Vista con Contornos', 
                 fontsize=12, fontweight='bold', pad=20)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=25, azim=45)
    ax.set_zlim(0, 1)

plt.tight_layout()
plt.savefig('memoria/imaxes/heatmaps/surface_plots_contour.png', dpi=300, bbox_inches='tight')
plt.show()