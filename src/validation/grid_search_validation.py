"""Grid search validation over bm25_weight and chunk_size with heatmap visualization."""
import argparse
import signal
import sys
import time
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from dotenv import load_dotenv
from tqdm import tqdm

from .shared_utils import (
    load_dataset,
    setup_llms,
    setup_rag_system,
    RAGEvaluator
)

# Global variable to track graceful shutdown
INTERRUPTED = False
CSV_PATH = None
PLOT_PATH = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global INTERRUPTED
    print("\n\n🛑 Interrupción detectada (Ctrl+C)...")
    print("Intentando crear visualizaciones con los datos disponibles...\n")
    INTERRUPTED = True


def evaluate_single_config(
    bm25_weight: float,
    chunk_size: int,
    dataset: list,
    base_config: dict,
    llm_judge,
    verbose: bool = False
) -> dict:
    """
    Evalúa una configuración específica de bm25_weight y chunk_size.
    """
    # Vectorstore único para este chunk_size
    vectorstore_path = base_config['output_dir'] / f"vectorstore_c{chunk_size}"
    
    # Configurar RAG
    rag = setup_rag_system(
        vectorstore_path=str(vectorstore_path),
        text_folder=base_config['text_folder'],
        state_dir=base_config['state_dir'],
        chunk_size=chunk_size,
        bm25_weight=bm25_weight,
        bm25_mode=base_config['bm25_mode'],
        k=base_config['k'],
        threshold=base_config['threshold'],
        language=base_config['language'],
        llm=base_config['llm'],
        llm_query=base_config['llm_query'],
        provider=base_config['provider'],
        use_bm25=True,
        verbose=verbose
    )
    
    if rag is None:
        print(f'⚠️  RAG initialization failed for chunk_size={chunk_size}, bm25_weight={bm25_weight}')
        return {
            'avg_recall': 0.0,
            'avg_precision': 0.0,
            'avg_mrr': 0.0,
            'avg_faithfulness': 0.0,
            'avg_relevance': 0.0,
            'f1_score': 0.0,
            'total_queries': len(dataset),
            'evaluated_queries': 0,
            'bm25_weight': bm25_weight,
            'chunk_size': chunk_size
        }
    
    # Crear evaluador (sin verbose para grid search)
    evaluator = RAGEvaluator(
        rag_system=rag,
        llm_judge=llm_judge,
        verbose=False  # Desactivar verbose del evaluador
    )
    
    # Evaluar dataset (sin guardar resultados, sin progreso interno)
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        k=base_config['k'],
        save_results=False,
        show_progress=False  # Sin barra de progreso interna
    )

    #time.sleep(1)
    
    # Extraer métricas agregadas
    metrics = results['aggregated_metrics'].copy()
    metrics['bm25_weight'] = bm25_weight
    metrics['chunk_size'] = chunk_size
    metrics['evaluated_queries'] = metrics.pop('total_questions')
    metrics['total_queries'] = len(dataset)
    
    return metrics


def run_grid_search(
    dataset_path: str,
    bm25_weights: list,
    chunk_sizes: list,
    base_config: dict,
    llm_judge,
    csv_path: Path,
    max_questions: int = None
) -> pd.DataFrame:
    """Ejecuta grid search con guardado incremental y barra de progreso."""
    
    dataset = load_dataset(dataset_path)
    
    # Limitar dataset si se especifica
    if max_questions is not None and max_questions < len(dataset):
        dataset = dataset[:max_questions]
        print(f"📊 Dataset limitado a {max_questions} preguntas")
    
    # Crear todas las combinaciones
    combinations = list(product(bm25_weights, chunk_sizes))
    total = len(combinations)
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {total} configuraciones")
    print(f"  BM25 weights: {bm25_weights}")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Dataset: {len(dataset)} preguntas")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}\n")
    
    # Verificar si ya existe el CSV y cargar resultados previos
    if csv_path.exists():
        print(f"⚠️  CSV existente encontrado, continuando desde donde se dejó...")
        df_existing = pd.read_csv(csv_path)
        completed_configs = set(
            zip(df_existing['bm25_weight'], df_existing['chunk_size'])
        )
        print(f"✅ {len(completed_configs)} configuraciones ya completadas\n")
    else:
        completed_configs = set()
        # Crear CSV con headers
        pd.DataFrame(columns=[
            'avg_recall', 'avg_precision', 'avg_mrr', 'avg_faithfulness',
            'avg_relevance', 'f1_score', 'total_queries', 'evaluated_queries',
            'bm25_weight', 'chunk_size'
        ]).to_csv(csv_path, index=False)
    
    # Filtrar combinaciones ya completadas
    combinations_to_run = [
        (w, c) for w, c in combinations 
        if (w, c) not in completed_configs
    ]
    
    if not combinations_to_run:
        print("✅ Todas las configuraciones ya están completadas!")
        return pd.read_csv(csv_path)
    
    print(f"🔄 Configuraciones pendientes: {len(combinations_to_run)}/{total}\n")
    
    # Usar tqdm para mostrar progreso
    with tqdm(combinations_to_run, desc="Grid Search Progress", unit="config") as pbar:
        for bm25_weight, chunk_size in pbar:
            if INTERRUPTED:
                print("\n🛑 Deteniendo evaluación...")
                break
            
            # Actualizar descripción de la barra
            pbar.set_description(
                f"bm25={bm25_weight:.2f}, chunk={chunk_size}"
            )
            
            try:
                # Evaluar esta configuración
                metrics = evaluate_single_config(
                    bm25_weight=bm25_weight,
                    chunk_size=chunk_size,
                    dataset=dataset,
                    base_config=base_config,
                    llm_judge=llm_judge,
                    verbose=False
                )
                
                # Guardar inmediatamente en CSV (append mode)
                df_row = pd.DataFrame([metrics])
                df_row.to_csv(csv_path, mode='a', header=False, index=False)
                
                # Actualizar postfix con métricas clave
                pbar.set_postfix({
                    'recall': f"{metrics['avg_recall']:.3f}",
                    'faith': f"{metrics['avg_faithfulness']:.3f}",
                    'mrr': f"{metrics['avg_mrr']:.3f}"
                })
                
            except KeyboardInterrupt:
                print("\n🛑 Interrupción detectada...")
                break
            except Exception as e:
                print(f"\n❌ Error en config (bm25={bm25_weight}, chunk={chunk_size}): {e}")
                # Guardar fila con ceros en caso de error
                error_metrics = {
                    'avg_recall': 0.0,
                    'avg_precision': 0.0,
                    'avg_mrr': 0.0,
                    'avg_faithfulness': 0.0,
                    'avg_relevance': 0.0,
                    'f1_score': 0.0,
                    'total_queries': len(dataset),
                    'evaluated_queries': 0,
                    'bm25_weight': bm25_weight,
                    'chunk_size': chunk_size
                }
                df_row = pd.DataFrame([error_metrics])
                df_row.to_csv(csv_path, mode='a', header=False, index=False)
                continue
    
    # Leer y retornar todo el CSV
    return pd.read_csv(csv_path)


def create_heatmaps(df: pd.DataFrame, output_path: Path):
    """Crea heatmaps de los resultados."""
    
    if df.empty:
        print("❌ No hay resultados para plotear.")
        return
    
    # Verificar que hay datos válidos (no todo ceros)
    if df[['avg_recall', 'avg_precision', 'avg_mrr', 'avg_faithfulness', 'avg_relevance']].sum().sum() == 0:
        print("⚠️  Todos los valores son 0, heatmap no será útil pero se generará de todos modos...")
    
    # Colormap personalizado
    colors = ['#ffffff', '#fde0e6', '#fbb4c4', '#f890a8', '#f56c8c', 
              '#f24870', '#ef2454', '#ec0038', '#c4002a']
    cmap = LinearSegmentedColormap.from_list('julia_pink', colors, N=256)
    
    metrics_to_plot = [
        'avg_recall', 
        'avg_precision', 
        'avg_mrr', 
        'avg_faithfulness', 
        'avg_relevance'
    ]
    
    # Crear figura
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, metric in enumerate(metrics_to_plot):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        
        # Pivot table
        pivot = df.pivot_table(
            values=metric,
            index='chunk_size',
            columns='bm25_weight',
            aggfunc='mean'
        )
        
        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=1
        )
        
        ax.set_title(
            f'{metric.replace("_", " ").title()}', 
            fontsize=14, 
            fontweight='bold'
        )
        ax.set_xlabel('BM25 Weight', fontsize=12)
        ax.set_ylabel('Chunk Size', fontsize=12)
    
    plt.suptitle(
        'Grid Search Results: BM25 Weight vs Chunk Size', 
        fontsize=18, 
        fontweight='bold', 
        y=0.98
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmaps guardados en: {output_path}")
    plt.close()


def print_best_configs(df: pd.DataFrame):
    """Imprime las mejores configuraciones encontradas."""
    
    if len(df) == 0:
        print("❌ No hay configuraciones para analizar")
        return
    
    # Filtrar filas con valores válidos (no todo ceros)
    df_valid = df[df['avg_recall'] > 0]
    
    if len(df_valid) == 0:
        print("⚠️  No hay configuraciones con resultados válidos (todas tienen métricas en 0)")
        return
    
    print(f"\n{'='*60}")
    print("MEJORES CONFIGURACIONES")
    print(f"{'='*60}")
    
    metrics_of_interest = [
        ('avg_recall', 'Recall'),
        ('avg_faithfulness', 'Faithfulness'),
        ('avg_mrr', 'MRR'),
        ('f1_score', 'F1-Score')
    ]
    
    for metric_key, metric_name in metrics_of_interest:
        if metric_key not in df_valid.columns:
            continue
            
        best = df_valid.loc[df_valid[metric_key].idxmax()]
        
        print(f"\n🏆 Best {metric_name}:")
        print(f"   BM25 Weight: {best['bm25_weight']:.2f}")
        print(f"   Chunk Size: {int(best['chunk_size'])}")
        print(f"   {metric_name}: {best[metric_key]:.3f}")
    
    print(f"\n{'='*60}\n")


def main():
    global CSV_PATH, PLOT_PATH
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description='Grid search validation over bm25_weight and chunk_size'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='validation/dataset_test.json'
    )
    
    parser.add_argument(
        '--questions', '-q',
        type=int,
        default=None,
        help='Limit dataset to first N questions (default: use all questions)'
    )
    
    # Grid search parameters
    parser.add_argument(
        '--bm25_weights', 
        type=str, 
        default='0.0, 0.2, 0.4, 0.8, 1.2, 1.6',
        help='Comma-separated BM25 weights'
    )
    parser.add_argument(
        '--chunk_sizes', 
        type=str, 
        default='256, 512, 1024, 2048',
        help='Comma-separated chunk sizes'
    )
    
    # Base configuration
    parser.add_argument('--text_folder', type=str, default='validation/validation_state/text')
    parser.add_argument('--state_dir', type=str, default='validation/validation_state')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--bm25_mode', type=str, default='hybrid',
                       choices=['bm25', 'hybrid', 'rerank'])
    
    # LLM
    parser.add_argument('--provider', type=str, default='anthropic')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022')
    parser.add_argument('--judge_provider', type=str, default='openai')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--language', type=str, default='galician')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='memoria/imaxes')
    parser.add_argument('--output_file', type=str, default='grid_search_results.csv')
    parser.add_argument('--plot_file', type=str, default='grid_search_heatmaps.png')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Parse parameters
    bm25_weights = [float(x.strip()) for x in args.bm25_weights.split(',')]
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global paths for signal handler
    CSV_PATH = output_dir / args.output_file
    PLOT_PATH = output_dir / args.plot_file
    
    try:
        # Setup LLMs (reutilizando función de validate_rag)
        print("\n🔧 Inicializando LLMs...")
        llm, llm_query, llm_judge = setup_llms(
            provider=args.provider,
            model=args.model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model
        )
        print("✅ LLMs inicializados")
        
        # Configuración base
        base_config = {
            'text_folder': args.text_folder,
            'state_dir': args.state_dir,
            'k': args.k,
            'threshold': args.threshold,
            'bm25_mode': args.bm25_mode,
            'language': args.language,
            'llm': llm,
            'llm_query': llm_query,
            'provider': args.provider,
            'output_dir': output_dir
        }
        
        # Run grid search
        df = run_grid_search(
            dataset_path=args.dataset,
            bm25_weights=bm25_weights,
            chunk_sizes=chunk_sizes,
            base_config=base_config,
            llm_judge=llm_judge,
            csv_path=CSV_PATH,
            max_questions=args.questions
        )
        
        print(f"\n✅ Resultados guardados en: {CSV_PATH}")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupción durante inicialización")
        if not CSV_PATH.exists():
            print("❌ No hay datos para procesar")
            sys.exit(1)
        df = pd.read_csv(CSV_PATH)
    
    # Crear heatmaps (siempre, incluso si hubo interrupción)
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if len(df) > 0:
            print(f"\n📊 Creando visualizaciones con {len(df)} configuraciones...")
            create_heatmaps(df, PLOT_PATH)
            print_best_configs(df)
        else:
            print("❌ CSV vacío, no se pueden crear visualizaciones")
    
    if INTERRUPTED:
        print("\n⚠️  Grid search interrumpido. Resultados parciales guardados.")
        sys.exit(130)  # Standard exit code for SIGINT
    else:
        print("\n✅Grid search completado exitosamente!")


if __name__ == "__main__":
    main()
