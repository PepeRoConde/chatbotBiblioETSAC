"""
Análisis de fechas Last-Modified en metadata.json
Genera gráficas para visualizar la distribución temporal de documentos
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configurar matplotlib para mostrar español
rcParams['font.family'] = 'sans-serif'

def analyze_last_modified():
    """Analiza las fechas Last-Modified en metadata.json"""
    
    metadata_path = Path("crawl/metadata.json")
    
    if not metadata_path.exists():
        print("❌ No se encontró crawl/metadata.json")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Contadores
    total_docs = len(metadata)
    with_last_modified = 0
    without_last_modified = 0
    dates_by_month = defaultdict(int)
    dates_by_year = defaultdict(int)
    
    print("\n📊 ANÁLISIS DE LAST-MODIFIED EN METADATA")
    print("=" * 70)
    
    for url, meta in metadata.items():
        if not isinstance(meta, dict):
            continue
        
        last_modified = meta.get("last_modified")
        
        if last_modified:
            with_last_modified += 1
            
            # Parsear fecha
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(last_modified)
                
                # Agrupar por mes
                month_key = dt.strftime("%Y-%m")
                dates_by_month[month_key] += 1
                
                # Agrupar por año
                year_key = dt.strftime("%Y")
                dates_by_year[year_key] += 1
                
            except Exception as e:
                print(f"⚠️  Error parseando fecha de {url}: {e}")
        else:
            without_last_modified += 1
    
    # Mostrar resumen
    print(f"\n📈 RESUMEN:")
    print(f"   Total documentos: {total_docs}")
    print(f"   Con Last-Modified: {with_last_modified} ({with_last_modified/total_docs*100:.1f}%)")
    print(f"   Sin Last-Modified: {without_last_modified} ({without_last_modified/total_docs*100:.1f}%)")
    
    if dates_by_year:
        print(f"\n📅 DISTRIBUCIÓN POR AÑO:")
        for year in sorted(dates_by_year.keys()):
            count = dates_by_year[year]
            bar = "█" * int(count / max(dates_by_year.values()) * 30)
            print(f"   {year}: {count:3d} {bar}")
    
    if dates_by_month:
        print(f"\n📆 DISTRIBUCIÓN POR MES (últimos meses):")
        recent_months = sorted(dates_by_month.keys())[-6:]  # Últimos 6 meses
        for month in recent_months:
            count = dates_by_month[month]
            bar = "█" * int(count / max(dates_by_month.values()) * 30)
            print(f"   {month}: {count:3d} {bar}")
    
    print("\n" + "=" * 70)
    
    # Generar gráficas
    generate_charts(total_docs, with_last_modified, without_last_modified, 
                   dates_by_year, dates_by_month)


def generate_charts(total, with_date, without_date, by_year, by_month):
    """Genera gráficas visuales"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Fechas Last-Modified en Documentos', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Pie chart - Con vs Sin Last-Modified
    ax1 = axes[0, 0]
    labels = [f'Con Last-Modified\n({with_date})', f'Sin Last-Modified\n({without_date})']
    sizes = [with_date, without_date]
    colors = ['#4CAF50', '#FF5252']
    explode = (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Disponibilidad de Last-Modified', fontsize=12, fontweight='bold')
    
    # Gráfico 2: Barras - Distribución por año
    ax2 = axes[0, 1]
    if by_year:
        years = sorted(by_year.keys())
        counts = [by_year[y] for y in years]
        
        bars = ax2.bar(years, counts, color='#2196F3', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Año', fontsize=11)
        ax2.set_ylabel('Cantidad de documentos', fontsize=11)
        ax2.set_title('Documentos por Año (Last-Modified)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Añadir valores en barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Sin datos de año disponibles', 
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    # Gráfico 3: Línea temporal - Últimos 12 meses
    ax3 = axes[1, 0]
    if by_month:
        recent_months = sorted(by_month.keys())[-12:]  # Últimos 12 meses
        counts = [by_month[m] for m in recent_months]
        
        ax3.plot(range(len(recent_months)), counts, marker='o', linewidth=2, 
                markersize=8, color='#FF9800', markerfacecolor='#FFC107')
        ax3.set_xticks(range(len(recent_months)))
        ax3.set_xticklabels(recent_months, rotation=45, ha='right', fontsize=9)
        ax3.set_xlabel('Mes', fontsize=11)
        ax3.set_ylabel('Cantidad de documentos', fontsize=11)
        ax3.set_title('Evolución Temporal (últimos 12 meses)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Añadir valores
        for i, count in enumerate(counts):
            ax3.text(i, count, str(count), ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Sin datos mensuales disponibles', 
                ha='center', va='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # Gráfico 4: Tabla de estadísticas
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [
        ['Métrica', 'Valor'],
        ['Total documentos', f'{total}'],
        ['Con Last-Modified', f'{with_date} ({with_date/total*100:.1f}%)'],
        ['Sin Last-Modified', f'{without_date} ({without_date/total*100:.1f}%)'],
        ['', ''],
        ['Cobertura temporal', '✅ Buena' if with_date/total > 0.7 else '⚠️ Regular' if with_date/total > 0.4 else '❌ Baja'],
    ]
    
    if by_year:
        oldest_year = min(by_year.keys())
        newest_year = max(by_year.keys())
        stats_data.append(['Rango temporal', f'{oldest_year} - {newest_year}'])
    
    table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de tabla
    for i in range(len(stats_data)):
        if i == 0:  # Header
            table[(i, 0)].set_facecolor('#1976D2')
            table[(i, 1)].set_facecolor('#1976D2')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif i == 4:  # Separador
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#f0f0f0')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f9f9f9')
                table[(i, 1)].set_facecolor('#f9f9f9')
    
    ax4.set_title('Estadísticas Detalladas', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Guardar gráfica
    output_path = Path("last_modified_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Gráfica guardada en: {output_path}")
    
    # Mostrar
    plt.show()


if __name__ == "__main__":
    analyze_last_modified()
