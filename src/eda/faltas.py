import pandas as pd
import re

# Leer el CSV
df = pd.read_csv('palabras_etiquetadas.csv')

# Asumiendo que tienes una columna 'palabra' y otra 'es_falta' (True/False o 1/0)
palabras_con_falta = df[df['label'] == True]['word'].tolist()
# O si es 1/0: palabras_con_falta = df[df['es_falta'] == 1]['palabra'].tolist()

# Leer los archivos OCR
with open('ocr_imagenes.txt', 'r', encoding='utf-8') as f:
    texto_imagenes = f.read().lower()

with open('ocr_tablas.txt', 'r', encoding='utf-8') as f:
    texto_tablas = f.read().lower()

# Contar coincidencias
contador_imagenes = {}
contador_tablas = {}

for palabra in palabras_con_falta:
    palabra_lower = palabra.lower()
    # Buscar palabra completa (con word boundaries)
    count_img = len(re.findall(r'\b' + re.escape(palabra_lower) + r'\b', texto_imagenes))
    count_tab = len(re.findall(r'\b' + re.escape(palabra_lower) + r'\b', texto_tablas))
    
    if count_img > 0:
        contador_imagenes[palabra] = count_img
    if count_tab > 0:
        contador_tablas[palabra] = count_tab

print(f"Palabras con falta encontradas en ocr_imagenes.txt: {len(contador_imagenes)}")
print(f"Total de ocurrencias: {sum(contador_imagenes.values())}")
print(f"\nPalabras con falta encontradas en ocr_tablas.txt: {len(contador_tablas)}")
print(f"Total de ocurrencias: {sum(contador_tablas.values())}")