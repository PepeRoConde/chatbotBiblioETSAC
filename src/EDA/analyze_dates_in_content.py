"""
Analiza cuántas fechas se pueden extraer del contenido de los documentos
usando regex para cursos académicos, años recientes, etc.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def extract_dates_from_text(text: str) -> dict:
    """Extrae fechas del contenido usando regex"""
    
    # Regex para curso académico (ej: "Curso 2024-2025", "curso académico 2023/2024")
    academic_year_pattern = r'\bcurso\s+(?:académico\s+)?(\d{4})[-/](\d{4})\b'
    academic_years = re.findall(academic_year_pattern, text, re.IGNORECASE)
    
    # Regex para años recientes (2020-2029)
    recent_year_pattern = r'\b(202[0-9])\b'
    recent_years = re.findall(recent_year_pattern, text)
    
    # Regex para fechas completas (DD/MM/YYYY, DD-MM-YYYY)
    full_date_pattern = r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'
    full_dates = re.findall(full_date_pattern, text)
    
    # Regex para meses y años (enero 2024, enero de 2024)
    month_year_pattern = r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?(\d{4})\b'
    month_years = re.findall(month_year_pattern, text, re.IGNORECASE)
    
    return {
        'academic_years': academic_years,
        'recent_years': list(set(recent_years)),  # Unique
        'full_dates': full_dates,
        'month_years': month_years
    }

def analyze_content_dates(solo_sin_lastmod=False):
    """
    Analiza fechas en documentos. Si solo_sin_lastmod=True, solo analiza documentos con last_modified = null.
    """
    text_dir = Path("crawl/text")
    metadata_path = Path("crawl/metadata.json")
    
    if not text_dir.exists():
        print("❌ No se encontró crawl/text/")
        return
    
    if not metadata_path.exists():
        print("❌ No se encontró crawl/metadata.json")
        return
    
    # Cargar metadata para saber qué documentos NO tienen last_modified
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Identificar documentos sin last_modified
    docs_without_last_modified = set()
    for url, meta in metadata.items():
        if isinstance(meta, dict) and meta.get('last_modified') is None:
            text_path = meta.get('text_path')
            if text_path:
                docs_without_last_modified.add(Path(text_path).name)
    
    # Estadísticas globales
    total_files = 0
    files_with_academic_year = 0
    files_with_recent_years = 0
    files_with_full_dates = 0
    files_with_month_year = 0
    files_with_any_date = 0
    files_without_dates = []

    # Histograma: número de fechas por documento
    num_dates_hist = Counter()
    example_files_by_count = defaultdict(list)

    # Estadísticas para documentos sin last_modified
    no_lastmod_total = 0
    no_lastmod_with_dates = 0
    no_lastmod_examples = []

    # Contadores de años encontrados
    all_academic_years = Counter()
    all_recent_years = Counter()
    
    print("\n📅 ANÁLISIS DE FECHAS EN CONTENIDO DE DOCUMENTOS")
    print("=" * 80)
    print(f"\n🔍 Analizando archivos en {text_dir}...\n")
    
    for text_file in text_dir.glob("*.txt"):
        # Si solo queremos los que no tienen last_modified, filtrar aquí
        if solo_sin_lastmod and text_file.name not in docs_without_last_modified:
            continue
        total_files += 1
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Limitar a primeros 50KB para eficiencia
            if len(content) > 50000:
                content = content[:50000]

            dates = extract_dates_from_text(content)

            # Contar el total de fechas encontradas en el documento
            num_dates = (
                len(dates['academic_years'])
                + len(dates['recent_years'])
                + len(dates['full_dates'])
                + len(dates['month_years'])
            )
            num_dates_hist[num_dates] += 1
            if len(example_files_by_count[num_dates]) < 3:
                example_files_by_count[num_dates].append(text_file.name)

            has_date = num_dates > 0

            if dates['academic_years']:
                files_with_academic_year += 1
                for year1, year2 in dates['academic_years']:
                    all_academic_years[f"{year1}-{year2}"] += 1

            if dates['recent_years']:
                files_with_recent_years += 1
                for year in dates['recent_years']:
                    all_recent_years[year] += 1

            if dates['full_dates']:
                files_with_full_dates += 1

            if dates['month_years']:
                files_with_month_year += 1

            if has_date:
                files_with_any_date += 1
            else:
                files_without_dates.append(text_file.name)

            # Estadísticas específicas para docs sin last_modified
            if text_file.name in docs_without_last_modified:
                no_lastmod_total += 1
                if has_date:
                    no_lastmod_with_dates += 1
                    no_lastmod_examples.append({
                        'file': text_file.name,
                        'dates': dates
                    })
        except Exception as e:
            print(f"⚠️  Error leyendo {text_file.name}: {e}")

    # Mostrar histograma de número de fechas por documento
    print("\n📈 HISTOGRAMA: NÚMERO DE FECHAS POR DOCUMENTO")
    print("-" * 80)
    # Agrupar todos los documentos con 20 o más fechas
    grouped_hist = Counter()
    grouped_examples = defaultdict(list)
    for count in num_dates_hist:
        if count < 20:
            grouped_hist[count] = num_dates_hist[count]
            grouped_examples[count] = example_files_by_count[count]
        else:
            grouped_hist[20] += num_dates_hist[count]
            grouped_examples[20].extend(example_files_by_count[count])

    for count in sorted(grouped_hist):
        label = f"{count} fechas" if count < 20 else "20 o más fechas"
        print(f"{label}: {grouped_hist[count]} documentos")
        ejemplos = grouped_examples[count]
        if ejemplos:
            print(f"   Ejemplos: {', '.join(ejemplos[:3])}")

    # Generar gráfica de barras
    try:
        x = [c for c in sorted(grouped_hist) if c < 20]
        y = [grouped_hist[k] for k in x]
        labels = [str(c) for c in x]
        if 20 in grouped_hist:
            x.append(20)
            y.append(grouped_hist[20])
            labels.append('20+')
        plt.figure(figsize=(8,5))
        # Color RGB (180,59,134) normalizado a [0,1]
        custom_color = (180/255, 59/255, 134/255)
        plt.bar(labels, y, color=custom_color)
        plt.xlabel('Número de fechas en documento')
        plt.ylabel('Número de documentos')
        plt.title('Histograma: Fechas extraídas por documento')
        plt.tight_layout()
        # Guardar en memoria/imaxes
        import os
        output_dir = os.path.join('memoria', 'imaxes')
        os.makedirs(output_dir, exist_ok=True)
        if solo_sin_lastmod:
            filename = 'histograma_fechas_por_documento_sin_lastmod.png'
        else:
            filename = 'histograma_fechas_por_documento.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        print(f"\n📊 Gráfica guardada como '{output_path}'")
    except Exception as e:
        print(f"⚠️  No se pudo generar la gráfica: {e}")
    
    # Mostrar resultados
    print("\n📊 ESTADÍSTICAS GENERALES:")
    print("-" * 80)
    print(f"Total archivos analizados: {total_files}")
    print(f"Con alguna fecha detectada: {files_with_any_date} ({files_with_any_date/total_files*100:.1f}%)")
    print(f"Sin fechas detectadas: {len(files_without_dates)} ({len(files_without_dates)/total_files*100:.1f}%)")
    
    print("\n📅 TIPOS DE FECHAS ENCONTRADAS:")
    print("-" * 80)
    print(f"Cursos académicos (ej: 'Curso 2024-2025'): {files_with_academic_year} ({files_with_academic_year/total_files*100:.1f}%)")
    print(f"Años recientes (2020-2029): {files_with_recent_years} ({files_with_recent_years/total_files*100:.1f}%)")
    print(f"Fechas completas (DD/MM/YYYY): {files_with_full_dates} ({files_with_full_dates/total_files*100:.1f}%)")
    print(f"Mes y año (enero 2024): {files_with_month_year} ({files_with_month_year/total_files*100:.1f}%)")
    
    print("\n🎓 CURSOS ACADÉMICOS MÁS MENCIONADOS:")
    print("-" * 80)
    for academic_year, count in all_academic_years.most_common(10):
        print(f"  {academic_year}: {count} documentos")
    
    print("\n📆 AÑOS MÁS MENCIONADOS:")
    print("-" * 80)
    for year, count in all_recent_years.most_common(10):
        print(f"  {year}: {count} documentos")
    
    print("\n⚠️  ANÁLISIS DE DOCUMENTOS SIN LAST-MODIFIED:")
    print("=" * 80)
    print(f"Total documentos sin Last-Modified: {len(docs_without_last_modified)}")
    print(f"De esos, tienen fechas en contenido: {no_lastmod_with_dates} ({no_lastmod_with_dates/max(no_lastmod_total,1)*100:.1f}%)")
    print(f"Sin Last-Modified NI fechas en contenido: {no_lastmod_total - no_lastmod_with_dates}")
    
    if no_lastmod_examples:
        print("\n📝 EJEMPLOS DE DOCS SIN LAST-MODIFIED PERO CON FECHAS EN CONTENIDO:")
        for i, example in enumerate(no_lastmod_examples[:5], 1):
            print(f"\n  {i}. {example['file']}")
            dates = example['dates']
            if dates['academic_years']:
                print(f"     Cursos: {dates['academic_years'][:3]}")
            if dates['recent_years']:
                print(f"     Años: {dates['recent_years'][:5]}")
    
    print("\n💡 CONCLUSIÓN:")
    print("=" * 80)
    coverage = files_with_any_date / total_files * 100
    if coverage > 70:
        print(f"✅ Muy buena cobertura ({coverage:.1f}%) - El regex VALE LA PENA")
        print("   Recomendación: Aplicar extracción de fechas del contenido")
    elif coverage > 40:
        print(f"⚠️  Cobertura moderada ({coverage:.1f}%) - Puede ser útil")
        print("   Recomendación: Aplicar solo para docs sin Last-Modified")
    else:
        print(f"❌ Baja cobertura ({coverage:.1f}%) - Regex NO es suficiente")
        print("   Recomendación: Dejar documentos sin fecha como null")
    
    # Calcular impacto en los 76 documentos sin last_modified
    if no_lastmod_total > 0:
        rescue_rate = no_lastmod_with_dates / no_lastmod_total * 100
        print(f"\n🎯 Para los {no_lastmod_total} docs sin Last-Modified:")
        print(f"   Podríamos rescatar fechas de {no_lastmod_with_dates} ({rescue_rate:.1f}%)")
        print(f"   Quedarían sin fecha: {no_lastmod_total - no_lastmod_with_dates}")

if __name__ == "__main__":
    import sys
    solo_null = False
    if len(sys.argv) > 1 and sys.argv[1] == "--solo-sin-lastmod":
        solo_null = True
    analyze_content_dates(solo_sin_lastmod=solo_null)
