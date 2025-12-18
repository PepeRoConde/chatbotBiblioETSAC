import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def detect_and_remove_pauses(metadata_path='crawl/metadata.json', gap_threshold_minutes=5):
    """
    Detecta pausas en el crawleo y ajusta las fechas para eliminarlas.
    
    Args:
        metadata_path: Ruta al archivo metadata.json
        gap_threshold_minutes: Tiempo mínimo (en minutos) para considerar una pausa
    """
    
    # Leer metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extraer fechas con sus URLs
    crawl_data = []
    for url, data in metadata.items():
        if 'last_crawl' in data and data['last_crawl']:
            try:
                date = datetime.fromisoformat(data['last_crawl'].replace('Z', '+00:00'))
                crawl_data.append({
                    'url': url,
                    'date': date,
                    'original_date': date
                })
            except (ValueError, AttributeError):
                continue
    
    if not crawl_data:
        print("No se encontraron fechas válidas")
        return
    
    # Ordenar por fecha
    crawl_data.sort(key=lambda x: x['date'])
    
    # Detectar pausas
    pauses = []
    gap_threshold = timedelta(minutes=gap_threshold_minutes)
    
    for i in range(1, len(crawl_data)):
        gap = crawl_data[i]['date'] - crawl_data[i-1]['date']
        if gap > gap_threshold:
            pauses.append({
                'index': i,
                'start': crawl_data[i-1]['date'],
                'end': crawl_data[i]['date'],
                'duration': gap
            })
    
    print(f"\n{'='*70}")
    print(f"Pausas detectadas (gaps > {gap_threshold_minutes} minutos):")
    print(f"{'='*70}")
    
    if not pauses:
        print("No se detectaron pausas significativas")
        return
    
    for idx, pause in enumerate(pauses, 1):
        print(f"\nPausa {idx}:")
        print(f"  Último doc antes: {pause['start'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Primer doc después: {pause['end'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duración: {pause['duration']}")
    
    print(f"\n{'='*70}")
    
    # Preguntar confirmación
    response = input("\n¿Quieres eliminar estas pausas del metadata? (s/n): ")
    if response.lower() != 's':
        print("Operación cancelada")
        return
    
    # Ajustar fechas eliminando las pausas
    total_adjustment = timedelta(0)
    
    for pause in pauses:
        pause_duration = pause['duration']
        total_adjustment += pause_duration
        
        # Ajustar todas las fechas después de esta pausa
        for item in crawl_data:
            if item['date'] >= pause['end']:
                item['date'] -= pause_duration
    
    # Actualizar metadata con las nuevas fechas
    for item in crawl_data:
        if item['date'] != item['original_date']:
            metadata[item['url']]['last_crawl'] = item['date'].isoformat()
    
    # Guardar backup
    import shutil
    backup_path = metadata_path + '.backup_before_pause_removal'
    shutil.copy2(metadata_path, backup_path)
    print(f"\nBackup creado: {backup_path}")
    
    # Guardar metadata ajustado
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Mostrar estadísticas
    print(f"\n{'='*70}")
    print("Resumen de ajustes:")
    print(f"{'='*70}")
    print(f"Número de pausas eliminadas: {len(pauses)}")
    print(f"Tiempo total eliminado: {total_adjustment}")
    print(f"Documentos ajustados: {sum(1 for item in crawl_data if item['date'] != item['original_date'])}")
    
    # Comparar antes/después
    original_duration = crawl_data[-1]['original_date'] - crawl_data[0]['original_date']
    adjusted_duration = crawl_data[-1]['date'] - crawl_data[0]['date']
    
    print(f"\nDuración original del crawleo: {original_duration}")
    print(f"Duración ajustada (sin pausas): {adjusted_duration}")
    print(f"Diferencia: {original_duration - adjusted_duration}")
    print(f"\n{'='*70}")
    print("\n✓ Metadata actualizado exitosamente")
    
    # Plotear comparación
    plot_comparison(crawl_data)

def plot_comparison(crawl_data):
    """Plotea comparación antes/después de eliminar pausas"""
    
    original_dates = [item['original_date'] for item in crawl_data]
    adjusted_dates = [item['date'] for item in crawl_data]
    doc_counts = list(range(1, len(crawl_data) + 1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfica original
    ax1.plot(original_dates, doc_counts, linewidth=2, color='#dc2626', label='Con pausas')
    ax1.set_xlabel('Tiempo', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Documentos crawleados', fontsize=11, fontweight='bold')
    ax1.set_title('ANTES: Crawleo con pausas', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Gráfica ajustada
    ax2.plot(adjusted_dates, doc_counts, linewidth=2, color='#16a34a', label='Sin pausas')
    ax2.set_xlabel('Tiempo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Documentos crawleados', fontsize=11, fontweight='bold')
    ax2.set_title('DESPUÉS: Crawleo sin pausas', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('crawl_pause_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGráfica comparativa guardada: crawl_pause_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Puedes ajustar el threshold de minutos para detectar pausas
    detect_and_remove_pauses(gap_threshold_minutes=5)