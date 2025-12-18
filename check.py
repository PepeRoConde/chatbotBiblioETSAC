import json
from pathlib import Path

def clean_orphaned_text_files(json_path='crawl/metadata.json', text_dir='crawl/text', dry_run=True):
    """Delete text files that are not referenced in metadata."""
    
    # Load metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Get all text_paths from metadata
    valid_text_files = set()
    for url, data in metadata.items():
        text_path = data.get('text_path')
        if text_path:
            # Extract just the filename
            filename = Path(text_path).name
            valid_text_files.add(filename)
    
    print(f"📝 Archivos válidos en metadata: {len(valid_text_files)}")
    
    # Get all actual text files
    text_path = Path(text_dir)
    if not text_path.exists():
        print(f"❌ La carpeta {text_dir} no existe")
        return
    
    actual_files = list(text_path.glob('*.txt'))
    print(f"📁 Archivos encontrados en {text_dir}: {len(actual_files)}")
    
    # Find orphaned files
    orphaned_files = []
    for file in actual_files:
        if file.name not in valid_text_files:
            orphaned_files.append(file)
    
    print(f"\n🗑️  Archivos huérfanos (no en metadata): {len(orphaned_files)}")
    
    if len(orphaned_files) == 0:
        print("✅ No hay archivos huérfanos para eliminar")
        return
    
    # Show some examples
    if len(orphaned_files) <= 10:
        print("\nArchivos a eliminar:")
        for file in orphaned_files:
            print(f"   - {file.name}")
    else:
        print(f"\nPrimeros 10 archivos a eliminar:")
        for file in orphaned_files[:10]:
            print(f"   - {file.name}")
        print(f"   ... y {len(orphaned_files) - 10} más")
    
    # Delete or dry run
    if dry_run:
        print(f"\n⚠️  MODO DRY RUN - No se eliminó nada")
        print(f"   Para eliminar realmente, ejecuta: clean_orphaned_text_files(dry_run=False)")
    else:
        deleted_count = 0
        for file in orphaned_files:
            try:
                file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error eliminando {file.name}: {e}")
        
        print(f"\n✅ Eliminados {deleted_count} archivos huérfanos")
    
    return {
        'valid_files': len(valid_text_files),
        'actual_files': len(actual_files),
        'orphaned': len(orphaned_files),
        'deleted': 0 if dry_run else deleted_count
    }

if __name__ == '__main__':
    # Primera ejecución: solo mostrar qué se eliminaría
    print("=== MODO DRY RUN ===\n")
    results = clean_orphaned_text_files(dry_run=False)
    
    # Para eliminar realmente, descomenta la siguiente línea:
    # print("\n=== ELIMINANDO ARCHIVOS ===\n")
    # results = clean_orphaned_text_files(dry_run=False)