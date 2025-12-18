import pandas as pd

# Leer el CSV
df = pd.read_csv('palabras_etiquetadas.csv')

# Sumar todos los valores de la columna 'label'
suma_total = df['label'].sum()

print(f"Suma total de la columna 'label': {suma_total}")