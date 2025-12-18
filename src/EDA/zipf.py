import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def leer_archivos(carpeta):
    """Lee todos los archivos .txt de la carpeta especificada"""
    textos = []
    longitudes = []
    
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.txt'):
            ruta = os.path.join(carpeta, archivo)
            with open(ruta, 'r', encoding='utf-8') as f:
                contenido = f.read()
                textos.append(contenido)
                longitudes.append(len(contenido))
    
    return textos, longitudes

def extraer_palabras(textos):
    """Extrae todas las palabras de los textos"""
    todas_palabras = []
    
    for texto in textos:
        # Convertir a minúsculas y extraer palabras (solo letras y números)
        palabras = re.findall(r'\b[a-záéíóúñü]+\b', texto.lower())
        todas_palabras.extend(palabras)
    
    return todas_palabras

def calcular_estadisticas(textos, palabras):
    """Calcula estadísticas del corpus"""
    # Caracteres totales
    total_caracteres = sum(len(texto) for texto in textos)
    
    # Palabras totales
    total_palabras = len(palabras)
    
    # Vocabulario (palabras únicas)
    vocabulario = set(palabras)
    tamano_vocabulario = len(vocabulario)
    
    # Frecuencia de palabras
    frecuencias = Counter(palabras)
    
    # Palabras con una sola ocurrencia
    palabras_unicas = [palabra for palabra, freq in frecuencias.items() if freq == 1]
    num_palabras_unicas = len(palabras_unicas)
    
    return {
        'total_caracteres': total_caracteres,
        'total_palabras': total_palabras,
        'tamano_vocabulario': tamano_vocabulario,
        'num_palabras_unicas': num_palabras_unicas,
        'palabras_unicas': sorted(palabras_unicas),
        'frecuencias': frecuencias
    }

def crear_visualizaciones(longitudes, frecuencias):
    """Crea histograma de longitudes y ley de Zipf"""
    # Calcular media y mediana
    media = np.mean(longitudes)
    mediana = np.median(longitudes)
    
    # Figura 1: Histograma y Zipf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma de longitudes (escala logarítmica en Y)
    ax1.hist(longitudes, bins=30, color="#E8759BE2", edgecolor='black', label='Distribución')
    
    # Añadir líneas de media y mediana
    ax1.axvline(media, color='gray', linestyle='--', linewidth=2, 
                label=f'Media: {media:.0f}')
    ax1.axvline(mediana, color='black', linestyle='--', linewidth=2, 
                label=f'Mediana: {mediana:.0f}')
    
    ax1.set_xlabel('Lonxitude (caracteres)')
    ax1.set_ylabel('Número de arquivos')
    ax1.set_title('Distribución da lonxitude de arquivos')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()
    
    # Ley de Zipf (vocabulario completo)
    frecuencias_ordenadas = sorted(frecuencias.values(), reverse=True)
    rangos = range(1, len(frecuencias_ordenadas) + 1)
    
    # Calcular frecuencia acumulada para encontrar el 80% del área
    total_palabras = sum(frecuencias_ordenadas)
    frecuencia_acumulada = np.cumsum(frecuencias_ordenadas)
    porcentaje_acumulado = frecuencia_acumulada / total_palabras
    
    # Encontrar dónde se alcanza el 80% de las ocurrencias
    indice_80_area = np.argmax(porcentaje_acumulado >= 0.80) + 1
    porcentaje_vocab_80 = (indice_80_area / len(frecuencias_ordenadas)) * 100
    
    # 20% del vocabulario
    indice_20_vocab = int(len(frecuencias_ordenadas) * 0.20)
    porcentaje_area_20 = porcentaje_acumulado[indice_20_vocab - 1] * 100
    
    ax2.loglog(rangos, frecuencias_ordenadas, marker='o', markersize=3, 
               linestyle='', color="#FFA0BF", alpha=1, label='Frecuencias')
    
    # Línea vertical en el 20% del vocabulario
    ax2.axvline(indice_20_vocab, color='gray', linestyle='--', linewidth=2, 
                label=f'20% vocab → {porcentaje_area_20:.1f}% ocorrencias')
    
    # Línea vertical donde se alcanza el 80% de las ocurrencias
    ax2.axvline(indice_80_area, color='black', linestyle='--', linewidth=2, 
                label=f'80% ocorrencias → {porcentaje_vocab_80:.1f}% vocab')
    
    ax2.set_xlabel('Rango (log)')
    ax2.set_ylabel('Frecuencia (log)')
    ax2.set_title('Ley de Zipf - Vocabulario completo')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('analisis_corpus.png', dpi=300, bbox_inches='tight')
    print("✓ Visualización guardada en 'analisis_corpus.png'")
    
    # Figura 2: Top 30 palabras más frecuentes
    fig2, ax3 = plt.subplots(figsize=(8, 10))
    palabras_top = frecuencias.most_common(30)
    palabras = [p[0] for p in palabras_top]
    frecuencias_vals = [p[1] for p in palabras_top]
    
    ax3.barh(range(len(palabras)), frecuencias_vals, color="#FF9FBF")
    ax3.set_yticks(range(len(palabras)))
    ax3.set_yticklabels(palabras, fontsize=10)
    ax3.set_xlabel('Frecuencia')
    ax3.set_title('Top 30 palabras mais frecuentes')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('top30_palabras.png', dpi=300, bbox_inches='tight')
    print("✓ Top 30 guardado en 'top30_palabras.png'")

def guardar_estadisticas(stats):
    """Guarda estadísticas generales en archivo"""
    with open('estadisticas_corpus.txt', 'w', encoding='utf-8') as f:
        f.write("ESTADÍSTICAS DEL CORPUS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total de caracteres: {stats['total_caracteres']:,}\n")
        f.write(f"Total de palabras: {stats['total_palabras']:,}\n")
        f.write(f"Tamaño del vocabulario: {stats['tamano_vocabulario']:,}\n")
        f.write(f"Palabras con una sola ocurrencia: {stats['num_palabras_unicas']:,}\n")
        f.write(f"Porcentaje de hapax legomena: {stats['num_palabras_unicas']/stats['tamano_vocabulario']*100:.2f}%\n")
    
    print("✓ Estadísticas guardadas en 'estadisticas_corpus.txt'")

def guardar_palabras_unicas(palabras_unicas):
    """Guarda palabras con una sola ocurrencia"""
    with open('palabras_unicas.txt', 'w', encoding='utf-8') as f:
        f.write("PALABRAS CON UNA SOLA OCURRENCIA\n")
        f.write("=" * 50 + "\n\n")
        for palabra in palabras_unicas:
            f.write(f"{palabra}\n")
    
    print("✓ Palabras únicas guardadas en 'palabras_unicas.txt'")

def main():
    carpeta = 'crawl/text'
    
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe")
        return
    
    print("Analizando archivos...")
    
    # Leer archivos
    textos, longitudes = leer_archivos(carpeta)
    print(f"✓ {len(textos)} archivos leídos")
    
    # Extraer palabras
    palabras = extraer_palabras(textos)
    print(f"✓ {len(palabras)} palabras extraídas")
    
    # Calcular estadísticas
    stats = calcular_estadisticas(textos, palabras)
    
    # Crear visualizaciones
    crear_visualizaciones(longitudes, stats['frecuencias'])
    
    # Guardar resultados
    guardar_estadisticas(stats)
    guardar_palabras_unicas(stats['palabras_unicas'])
    
    print("\n¡Análisis completado!")
    print(f"\nResumen:")
    print(f"  - Archivos procesados: {len(textos)}")
    print(f"  - Total caracteres: {stats['total_caracteres']:,}")
    print(f"  - Total palabras: {stats['total_palabras']:,}")
    print(f"  - Vocabulario: {stats['tamano_vocabulario']:,}")
    print(f"  - Palabras únicas: {stats['num_palabras_unicas']:,}")

if __name__ == "__main__":
    main()
