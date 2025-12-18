import json
import matplotlib.pyplot as plt
from collections import Counter

# Leer el archivo metadata.json
with open('crawl/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Contar formatos
formatos = []
for url, info in metadata.items():
    formato = info.get('original_format', 'unknown')
    formatos.append(formato)

# Contar ocurrencias
contador = Counter(formatos)

# Datos para el pie chart
labels = []
sizes = []
totales = []

for formato, count in contador.items():
    labels.append(formato.upper())
    sizes.append(count)
    totales.append(count)

# Calcular porcentajes
total_docs = sum(totales)
porcentajes = [(count / total_docs) * 100 for count in totales]

# Crear etiquetas con número y porcentaje
labels_completos = [f'{label}\n{count} ({pct:.1f}%)' 
                    for label, count, pct in zip(labels, totales, porcentajes)]

# Crear el pie chart
fig, ax = plt.subplots(figsize=(10, 8))

# Colores rosa UDC
colors = ['#E91E63', '#F06292', '#F8BBD0', '#FCE4EC']

wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels_completos,
    autopct='',  # No mostrar porcentaje adicional dentro
    startangle=90,
    colors=colors[:len(sizes)],
    textprops={'fontsize': 12, 'weight': 'bold'}
)

ax.set_title('Distribución de Formatos de Documentos', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('formatos_documentos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado en 'formatos_documentos.png'")

# Imprimir estadísticas
print("\n" + "="*50)
print("ESTADÍSTICAS DE FORMATOS")
print("="*50)
print(f"\nTotal de documentos: {total_docs}")
print("\nDesglose por formato:")
for formato, count in contador.most_common():
    porcentaje = (count / total_docs) * 100
    print(f"  {formato.upper()}: {count} ({porcentaje:.2f}%)")
print("="*50)