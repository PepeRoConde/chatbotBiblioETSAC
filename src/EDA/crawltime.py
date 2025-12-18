import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_crawl_progress(metadata_path='crawl/metadata.json', window_minutes=1):
    """
    Plotea la velocidad del crawleo a lo largo del tiempo:
    Eje X: Tiempo
    Eje Y: Documentos por minuto (ventana deslizante)
    """
    
    # Leer metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extraer fechas de last_crawl
    crawl_dates = []
    
    for url, data in metadata.items():
        if 'last_crawl' in data and data['last_crawl']:
            try:
                date = datetime.fromisoformat(data['last_crawl'].replace('Z', '+00:00'))
                crawl_dates.append(date)
            except (ValueError, AttributeError) as e:
                continue
    
    if not crawl_dates:
        print("No se encontraron fechas válidas en last_crawl")
        return
    
    # Ordenar fechas
    crawl_dates.sort()
    
    # Calcular velocidad con ventana deslizante
    window = timedelta(minutes=window_minutes)
    times = []
    rates = []
    
    for i, current_time in enumerate(crawl_dates):
        # Contar docs en la ventana [current_time - window, current_time]
        count = sum(1 for t in crawl_dates if current_time - window <= t <= current_time)
        docs_per_minute = count / window_minutes
        
        times.append(current_time)
        rates.append(docs_per_minute)
    
    # Calcular velocidad promedio (línea y=x equivalente)
    start_time = crawl_dates[0]
    end_time = crawl_dates[-1]
    duration_minutes = (end_time - start_time).total_seconds() / 60
    avg_rate = len(crawl_dates) / duration_minutes if duration_minutes > 0 else 0
    
    # Crear la gráfica
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Gráfica 1: Velocidad instantánea
    ax1.plot(times, rates, linewidth=1.5, color='#ec4899', alpha=0.7, label='Velocidade real')
    ax1.axhline(y=avg_rate, color='#db2777', linestyle='--', linewidth=2, 
                label=f'Velocidade media ({avg_rate:.2f} docs/min)', alpha=0.6)
    
    ax1.set_ylabel(f'Documentos por minuto (xanela {window_minutes} min)', 
                   fontsize=11, fontweight='bold')
    ax1.set_title('Velocidade do Crawleo ao Longo do Tempo', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Formatear eje X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Gráfica 2: Acumulado para contexto
    doc_counts = list(range(1, len(crawl_dates) + 1))
    ax2.plot(crawl_dates, doc_counts, linewidth=2, color='#f472b6', alpha=0.7)
    ax2.set_xlabel('Tempo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Docs acumulados', fontsize=11, fontweight='bold')
    ax2.set_title('Progreso Acumulado (contexto)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Formatear eje X
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.gcf().autofmt_xdate()
    
    # Estadísticas
    duration = end_time - start_time
    total_docs = len(crawl_dates)
    
    stats_text = f'Total: {total_docs} docs\n'
    stats_text += f'Duración: {duration}\n'
    stats_text += f'Promedio: {avg_rate:.2f} docs/min\n'
    stats_text += f'Máximo: {max(rates):.2f} docs/min\n'
    stats_text += f'Mínimo: {min(rates):.2f} docs/min'
    
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9,
             family='monospace')
    
    plt.tight_layout()
    
    # Guardar y mostrar
    plt.savefig('memoria/imaxes/crawl_speed.png', dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print("Análise de Velocidade do Crawleo:")
    print(f"{'='*60}")
    print(stats_text)
    print(f"{'='*60}")
    print("\nGráfica gardada como: crawl_speed.png")
    
    plt.show()

if __name__ == "__main__":
    # Puedes ajustar el tamaño de la ventana
    plot_crawl_progress(window_minutes=1)