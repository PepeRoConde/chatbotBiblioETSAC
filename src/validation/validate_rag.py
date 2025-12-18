"""
Script de evaluación del sistema RAG con visualizaciones mejoradas.
Métricas: Recall, MRR, Precision, Faithfulness (LLM as judge)
"""
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rich.console import Console
from rich.table import Table

# Import from shared utilities
from .shared_utils import (
    load_dataset,
    setup_llms,
    setup_rag_system,
    RAGEvaluator
)


def create_enhanced_visualizations(
    results: list,
    aggregated_metrics: dict,
    output_dir: str,
    config_name: str = ""
):
    """
    Crea visualizaciones mejoradas con seaborn.
    Incluye: distribuciones, comparaciones, y análisis detallado.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{config_name}" if config_name else ""
    
    # Preparar datos
    df_data = []
    for r in results:
        df_data.append({
            'question': r.question[:50] + '...' if len(r.question) > 50 else r.question,
            'recall': r.retrieval.recall,
            'precision': r.retrieval.precision,
            'mrr': r.retrieval.mrr,
            'rank_first_relevant': r.retrieval.rank_first_relevant,
            'faithfulness': r.generation.faithfulness_score,
            'relevance': r.generation.answer_relevance_score,
        })
    
    df = pd.DataFrame(df_data)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="husl")
    
    # ========== FIGURA 1: Distribuciones de Métricas ==========
    fig1 = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)
    
    metrics_to_plot = ['recall', 'precision', 'mrr', 'faithfulness', 'relevance']
    colors = sns.color_palette("husl", len(metrics_to_plot))
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = fig1.add_subplot(gs1[row, col])
        
        # Histogram + KDE
        sns.histplot(
            data=df,
            x=metric,
            kde=True,
            color=colors[idx],
            alpha=0.6,
            bins=15,
            ax=ax
        )
        
        # Add mean line
        mean_val = df[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.suptitle('RAG Metrics Distributions', fontsize=16, fontweight='bold', y=0.98)
    plot1_path = output_path / f"distributions{suffix}.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== FIGURA 2: Comparación de Métricas por Pregunta ==========
    fig2, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Retrieval Metrics
    df_retrieval = df[['recall', 'precision', 'mrr']].reset_index()
    df_retrieval_melted = df_retrieval.melt(
        id_vars='index', 
        var_name='Metric', 
        value_name='Score'
    )
    
    sns.lineplot(
        data=df_retrieval_melted,
        x='index',
        y='Score',
        hue='Metric',
        marker='o',
        ax=axes[0],
        linewidth=2,
        markersize=6
    )
    axes[0].set_title('Retrieval Metrics per Question', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Question Index', fontsize=10)
    axes[0].set_ylabel('Score', fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(title='Metric', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Generation Metrics
    df_generation = df[['faithfulness', 'relevance']].reset_index()
    df_generation_melted = df_generation.melt(
        id_vars='index',
        var_name='Metric',
        value_name='Score'
    )
    
    sns.lineplot(
        data=df_generation_melted,
        x='index',
        y='Score',
        hue='Metric',
        marker='s',
        ax=axes[1],
        linewidth=2,
        markersize=6
    )
    axes[1].set_title('Generation Metrics per Question', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Question Index', fontsize=10)
    axes[1].set_ylabel('Score', fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(title='Metric', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Rank of First Relevant Document
    sns.barplot(
        data=df.reset_index(),
        x='index',
        y='rank_first_relevant',
        ax=axes[2],
        palette='viridis'
    )
    axes[2].set_title('Rank of First Relevant Document per Question', 
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Question Index', fontsize=10)
    axes[2].set_ylabel('Rank', fontsize=10)
    axes[2].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Rank 5')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot2_path = output_path / f"per_question_metrics{suffix}.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== FIGURA 3: Correlaciones y Heatmap ==========
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation matrix
    correlation_cols = ['recall', 'precision', 'mrr', 'faithfulness', 'relevance']
    corr_matrix = df[correlation_cols].corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes[0],
        cbar_kws={'label': 'Correlation'}
    )
    axes[0].set_title('Metric Correlations', fontsize=12, fontweight='bold')
    
    # Pairplot style scatter
    # Choose two key metrics for scatter
    axes[1].scatter(
        df['recall'],
        df['faithfulness'],
        c=df['relevance'],
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    axes[1].set_xlabel('Recall', fontsize=10)
    axes[1].set_ylabel('Faithfulness', fontsize=10)
    axes[1].set_title('Recall vs Faithfulness (colored by Relevance)', 
                      fontsize=12, fontweight='bold')
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('Relevance', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = output_path / f"correlations{suffix}.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== FIGURA 4: Métricas Agregadas (Barplot) ==========
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    agg_data = {
        'Recall': aggregated_metrics['avg_recall'],
        'Precision': aggregated_metrics['avg_precision'],
        'MRR': aggregated_metrics['avg_mrr'],
        'F1-Score': aggregated_metrics['f1_score'],
        'Faithfulness': aggregated_metrics['avg_faithfulness'],
        'Relevance': aggregated_metrics['avg_relevance'],
        'Recall@1': aggregated_metrics['recall@1'],
        'Recall@5': aggregated_metrics['recall@5'],
        'Recall@10': aggregated_metrics['recall@10']
    }
    
    agg_df = pd.DataFrame(list(agg_data.items()), columns=['Metric', 'Score'])
    
    colors_palette = sns.color_palette("Set2", len(agg_df))
    bars = ax.barh(agg_df['Metric'], agg_df['Score'], color=colors_palette)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}',
            ha='left',
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Aggregated RAG Metrics', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot4_path = output_path / f"aggregated_metrics{suffix}.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[green]✓ Visualizaciones guardadas:[/green]")
    print(f"  - Distribuciones: {plot1_path}")
    print(f"  - Métricas por pregunta: {plot2_path}")
    print(f"  - Correlaciones: {plot3_path}")
    print(f"  - Agregados: {plot4_path}")


def print_summary_table(metrics: dict, console: Console):
    """Imprime tabla resumen de métricas con Rich."""
    
    table = Table(title="📊 MÉTRICAS DE EVALUACIÓN DEL SISTEMA RAG", show_header=True)
    table.add_column("Métrica", style="cyan", width=30)
    table.add_column("Valor", style="magenta", justify="right", width=15)
    table.add_column("Descripción", style="dim", width=40)
    
    table.add_row(
        "Recall promedio",
        f"{metrics['avg_recall']:.3f}",
        "% de docs relevantes recuperados"
    )
    table.add_row(
        "Precision promedio",
        f"{metrics['avg_precision']:.3f}",
        "% de docs recuperados relevantes"
    )
    table.add_row(
        "F1-Score",
        f"{metrics['f1_score']:.3f}",
        "Media armónica P y R"
    )
    table.add_row(
        "MRR (Mean Reciprocal Rank)",
        f"{metrics['avg_mrr']:.3f}",
        "Posición promedio del 1er relevante"
    )
    
    table.add_row("", "", "", style="dim")
    table.add_row(
        "Recall@1",
        f"{metrics['recall@1']:.3f}",
        "% con relevante en posición 1"
    )
    table.add_row(
        "Recall@5",
        f"{metrics['recall@5']:.3f}",
        "% con relevante en top-5"
    )
    table.add_row(
        "Recall@10",
        f"{metrics['recall@10']:.3f}",
        "% con relevante en top-10"
    )
    
    table.add_row("", "", "", style="dim")
    table.add_row(
        "Faithfulness promedio",
        f"{metrics['avg_faithfulness']:.3f}",
        "Fidelidad al contexto (LLM judge)"
    )
    table.add_row(
        "Relevance promedio",
        f"{metrics['avg_relevance']:.3f}",
        "Pertinencia de la respuesta (LLM judge)"
    )
    
    table.add_row("", "", "", style="dim")
    table.add_row(
        "Total de preguntas",
        f"{metrics['total_questions']}",
        ""
    )
    
    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Evaluar sistema RAG')
    
    # Dataset
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='validation/dataset_test.json',
        help='Archivo JSON con preguntas de evaluación'
    )
    
    parser.add_argument('--text_folder', type=str, default='validation/validation_state/text')
    parser.add_argument('--state_dir', type=str, default='validation/validation_state')
    parser.add_argument('--vector_store', type=str, default='validation/validation_vectorstore')
    parser.add_argument('--k', type=int, default=10, help='Documentos a recuperar')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--bm25_mode', type=str, default='bm25', 
                        choices=['bm25', 'hybrid', 'rerank'])
    parser.add_argument('--bm25_weight', type=float, default=0.3)
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--use_bm25', action='store_true', default=True)
    parser.add_argument('--no_bm25', action='store_false', dest='use_bm25')
   
    # LLM
    parser.add_argument('--provider', type=str, default='claude')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022')
    parser.add_argument('--judge_provider', type=str, default='openai')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--language', type=str, default='galician')
    
    # Output
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='validation/results',
        help='Directorio para guardar resultados'
    )
    
    parser.add_argument(
        '--create_plots',
        action='store_true',
        default=True,
        help='Crear visualizaciones con seaborn'
    )
    
    args = parser.parse_args()
    
    config_name = f"{args.bm25_mode}_chunk{args.chunk_size}"
    
    console = Console()
    
    # Cargar dataset
    console.print(f"\n[cyan]Cargando dataset de evaluación...[/cyan]")
    dataset = load_dataset(args.dataset)
    console.print(f"[green]✓ {len(dataset)} preguntas cargadas[/green]")
    
    # Inicializar LLMs
    console.print(f"\n[cyan]Inicializando LLMs ({args.provider})...[/cyan]")
    llm, llm_query, llm_judge = setup_llms(
        provider=args.provider,
        model=args.model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model
    )
    
    console.print(f"[green]✓ LLMs inicializados:[/green]")
    console.print(f"  - Query optimization: Haiku")
    console.print(f"  - Answer generation: {args.model}")
    console.print(f"  - Evaluation (judge): {args.judge_provider}/{args.judge_model}")
    console.print(f"[yellow]Configuración:[/yellow]")
    console.print(f"  - Chunk size: {args.chunk_size}")
    console.print(f"  - BM25 mode: {args.bm25_mode}")
    console.print(f"  - Use BM25: {args.use_bm25}")
    
    # Setup RAG
    console.print(f"\n[cyan]Configurando sistema RAG...[/cyan]")
    rag = setup_rag_system(
        vectorstore_path=args.vector_store,
        text_folder=args.text_folder,
        state_dir=args.state_dir,
        chunk_size=args.chunk_size,
        bm25_weight=args.bm25_weight,
        bm25_mode=args.bm25_mode,
        k=args.k,
        threshold=args.threshold,
        language=args.language,
        llm=llm,
        llm_query=llm_query,
        provider=args.provider,
        use_bm25=args.use_bm25,
        verbose=True
    )
    console.print(f"[green]✓ RAG inicializado[/green]")
    
    # Crear evaluador
    evaluator = RAGEvaluator(
        rag_system=rag,
        llm_judge=llm_judge,
        console=console,
        verbose=True
    )
    
    # Evaluar
    console.print(f"\n[bold green]Iniciando evaluación...[/bold green]\n")
    
    results_dict = evaluator.evaluate_dataset(
        dataset=dataset,
        k=args.k,
        save_results=True,
        output_dir=args.output_dir,
        config_name=config_name,
        show_progress=True
    )
    
    # Extraer resultados
    individual_results = results_dict['individual_results']
    aggregated_metrics = results_dict['aggregated_metrics']
    
    # Mostrar tabla resumen
    print_summary_table(aggregated_metrics, console)
    
    # Crear visualizaciones
    if args.create_plots:
        console.print(f"\n[cyan]Creando visualizaciones...[/cyan]")
        create_enhanced_visualizations(
            results=individual_results,
            aggregated_metrics=aggregated_metrics,
            output_dir=args.output_dir,
            config_name=config_name
        )
    
    console.print(f"\n[bold green]✓ Evaluación completada[/bold green]")


if __name__ == "__main__":
    main()
