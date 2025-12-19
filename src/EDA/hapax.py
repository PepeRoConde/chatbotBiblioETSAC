import csv
from spellchecker import SpellChecker

def leer_hapax(archivo):
    """Lee las palabras del archivo de hapax legomena"""
    palabras = []
    
    with open(archivo, 'r', encoding='utf-8') as f:
        # Saltar el encabezado (primeras 3 líneas)
        for _ in range(3):
            next(f)
        
        # Leer palabras
        for linea in f:
            palabra = linea.strip()
            if palabra:  # Ignorar líneas vacías
                palabras.append(palabra)
    
    return palabras

def verificar_ortografia(palabras):
    """Verifica la ortografía de cada palabra en español, inglés y gallego"""
    spell_es = SpellChecker(language='es')
    spell_en = SpellChecker(language='en')
    spell_gl = SpellChecker(language='gl')
    
    resultados = []
    for palabra in palabras:
        # Verificar en cada idioma
        error_es = 1 if palabra in spell_es.unknown([palabra]) else 0
        error_en = 1 if palabra in spell_en.unknown([palabra]) else 0
        error_gl = 1 if palabra in spell_gl.unknown([palabra]) else 0
        
        # Si está correcta en al menos un idioma, no es error
        es_error = 1 if (error_es and error_en and error_gl) else 0
        
        resultados.append({
            'palabra': palabra,
            'error_ortografico': es_error,
            'error_espanol': error_es,
            'error_ingles': error_en,
            'error_galego': error_gl
        })
    
    return resultados

def guardar_csv(resultados, archivo_salida):
    """Guarda los resultados en un archivo CSV"""
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'palabra', 'error_ortografico', 'error_espanol', 
            'error_ingles', 'error_galego'
        ])
        writer.writeheader()
        writer.writerows(resultados)

def main():
    archivo_entrada = 'palabras_unicas.txt'
    archivo_salida = 'hapax_ortografia.csv'
    
    print("Leyendo hapax legomena...")
    palabras = leer_hapax(archivo_entrada)
    print(f"✓ {len(palabras)} palabras leídas")
    
    print("\nVerificando ortografía en español, inglés y gallego...")
    resultados = verificar_ortografia(palabras)
    
    # Estadísticas
    errores = sum(1 for r in resultados if r['error_ortografico'] == 1)
    correctas = len(resultados) - errores
    
    print(f"✓ Verificación completada")
    print(f"  - Palabras correctas: {correctas} ({correctas/len(resultados)*100:.1f}%)")
    print(f"  - Posibles errores: {errores} ({errores/len(resultados)*100:.1f}%)")
    
    print(f"\nGuardando CSV...")
    guardar_csv(resultados, archivo_salida)
    print(f"✓ Resultados guardados en '{archivo_salida}'")

if __name__ == "__main__":
    main()