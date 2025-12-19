import os
import re

def extraer_ocr_y_tablas(carpeta):
    """Extrae texto OCR de imágenes y tablas de los archivos"""
    texto_imagenes = []
    texto_tablas = []
    
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.txt'):
            ruta = os.path.join(carpeta, archivo)
            with open(ruta, 'r', encoding='utf-8') as f:
                contenido = f.read()
            
            # Extraer imágenes OCR
            imagenes = extraer_imagenes_ocr(contenido)
            texto_imagenes.extend(imagenes)
            
            # Extraer tablas
            tablas = extraer_tablas(contenido)
            texto_tablas.extend(tablas)
    
    return texto_imagenes, texto_tablas

def extraer_imagenes_ocr(contenido):
    """Extrae texto de imágenes OCR y su contexto asociado"""
    imagenes = []
    
    # Patrón más flexible: busca cualquier línea que contenga "--- OCR"
    patron_imagen = r'--- OCR[^\n]*\n(.*?)(?=--- OCR|=== Taboa|$)'
    
    matches = re.finditer(patron_imagen, contenido, re.DOTALL)
    
    for i, match in enumerate(matches, 1):
        texto_ocr = match.group(1).strip()
        
        if texto_ocr:  # Solo si hay texto extraído
            # Intentar extraer el nombre de la imagen de la línea completa
            linea_completa = match.group(0).split('\n')[0]
            imagenes.append({
                'nombre': linea_completa.strip('- '),
                'texto': texto_ocr
            })
    
    return imagenes

def extraer_tablas(contenido):
    """Extrae texto de tablas y su contexto asociado"""
    tablas = []
    
    # Patrón: busca líneas que empiecen con === y captura hasta el siguiente === o --- OCR
    patron_tabla = r'^===\s*(.+?)\s*===\s*\n(.*?)(?=^===|^--- OCR|$)'
    
    matches = re.finditer(patron_tabla, contenido, re.DOTALL | re.MULTILINE)
    
    for i, match in enumerate(matches, 1):
        nombre_tabla = match.group(1).strip()
        texto_tabla = match.group(2).strip()
        
        if texto_tabla:  # Solo si hay texto extraído
            tablas.append({
                'numero': nombre_tabla,
                'texto': texto_tabla
            })
    
    return tablas

def guardar_resultados(imagenes, tablas):
    """Guarda los resultados en archivos separados"""
    
    # Guardar texto de imágenes OCR
    with open('ocr_imagenes.txt', 'w', encoding='utf-8') as f:
        f.write("TEXTO EXTRAÍDO POR OCR DE IMÁGENES\n")
        f.write("=" * 70 + "\n\n")
        
        for i, img in enumerate(imagenes, 1):
            f.write(f"[IMAGEN {i}] {img['nombre']}\n")
            f.write("-" * 70 + "\n")
            f.write(img['texto'])
            f.write("\n\n" + "=" * 70 + "\n\n")
    
    print(f"✓ {len(imagenes)} imágenes OCR guardadas en 'ocr_imagenes.txt'")
    
    # Guardar texto de tablas
    with open('ocr_tablas.txt', 'w', encoding='utf-8') as f:
        f.write("TEXTO EXTRAÍDO DE TABLAS\n")
        f.write("=" * 70 + "\n\n")
        
        for tabla in tablas:
            f.write(f"[TABLA {tabla['numero']}]\n")
            f.write("-" * 70 + "\n")
            f.write(tabla['texto'])
            f.write("\n\n" + "=" * 70 + "\n\n")
    
    print(f"✓ {len(tablas)} tablas guardadas en 'ocr_tablas.txt'")

def main():
    carpeta = 'crawl/text'
    
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta '{carpeta}' no existe")
        return
    
    print("Extrayendo texto OCR y tablas...")
    
    # Extraer imágenes y tablas
    imagenes, tablas = extraer_ocr_y_tablas(carpeta)
    
    # Guardar resultados
    guardar_resultados(imagenes, tablas)
    
    print("\n¡Extracción completada!")
    print(f"\nResumen:")
    print(f"  - Imágenes OCR encontradas: {len(imagenes)}")
    print(f"  - Tablas encontradas: {len(tablas)}")

if __name__ == "__main__":
    main()