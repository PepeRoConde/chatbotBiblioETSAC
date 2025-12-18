import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('crawl/ocr_stats.csv')

# Opción 1: Contar TRUE en una columna específica
columna = 'is_table'  # Cambia por el nombre de tu columna
num_trues = df[columna].sum()  

print(f"Número de TRUE en '{columna}': {num_trues}")

