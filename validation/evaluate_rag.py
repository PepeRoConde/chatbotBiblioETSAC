"""
Script de evaluación del sistema RAG.
Métricas: Recall, MRR, Precision, Faithfulness (LLM as judge)
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from RAGSystem import RAGSystem
from LLMManager import LLMManager


@dataclass
class RetrievalMetrics:
    """Métricas de recuperación para una pregunta."""
    question: str
    gold_docs: List[str]
    retrieved_docs: List[str]
    recall: float
    precision: float
    mrr: float  # Mean Reciprocal Rank
    rank_first_relevant: int  # Posición del primer doc relevante


@dataclass
class GenerationMetrics:
    """Métricas de generación para una pregunta."""
    question: str
    generated_answer: str
    expected_answers: List[str]
    faithfulness_score: float  # 0-1, evaluado por LLM
    faithfulness_explanation: str


@dataclass
class QuestionEvaluation:
    """Evaluación completa de una pregunta."""
    question: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    retrieved_contexts: List[str]


class RAGEvaluator:
    """Evaluador del sistema RAG."""
    
    def __init__(
        self,
        rag_system: RAGSystem,
        llm_judge: Any,  # LLM para evaluar faithfulness
        console: Console = None
    ):
        self.rag = rag_system
        self.llm_judge = llm_judge
        self.console = console or Console()
    
    def calculate_retrieval_metrics(
        self,
        question: str,
        gold_docs: List[str],
        retrieved_docs: List[str]
    ) -> RetrievalMetrics:
        """Calcula métricas de recuperación."""
        
        # Normalizar nombres de documentos (comparar sin extensiones ni rutas)
        def normalize_doc_name(doc: str) -> str:
            name = Path(doc).name.lower()  # Obtener solo el nombre del archivo
            # Eliminar todas las extensiones .txt (puede haber múltiples)
            while name.endswith('.txt'):
                name = name[:-4]
            return name
        
        gold_normalized = {normalize_doc_name(d) for d in gold_docs}
        retrieved_normalized = [normalize_doc_name(d) for d in retrieved_docs]
        
        # RECALL: Documentos únicos relevantes (entre 0 y 1)
        # "De todos los documentos gold, ¿cuántos únicos encontré en el Top-K?"
        retrieved_unique = set(retrieved_normalized)
        relevant_docs_found = len(gold_normalized.intersection(retrieved_unique))
        recall = relevant_docs_found / len(gold_normalized) if gold_normalized else 0.0
        
        # PRECISION: A nivel chunk
        # "De todos los chunks recuperados, ¿cuántos son relevantes?"
        relevant_chunks = sum(1 for d in retrieved_normalized if d in gold_normalized)
        precision = relevant_chunks / len(retrieved_normalized) if retrieved_normalized else 0.0
        
        # MRR: Mean Reciprocal Rank
        # Encontrar posición del primer documento relevante
        rank_first_relevant = -1
        mrr = 0.0
        for idx, doc in enumerate(retrieved_normalized, 1):
            if doc in gold_normalized:
                rank_first_relevant = idx
                mrr = 1.0 / idx
                break
        
        return RetrievalMetrics(
            question=question,
            gold_docs=gold_docs,
            retrieved_docs=retrieved_docs,
            recall=recall,
            precision=precision,
            mrr=mrr,
            rank_first_relevant=rank_first_relevant
        )
    
    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        max_retries: int = 2
    ) -> Tuple[float, str]:
        """
        Evalúa la fidelidad de la respuesta usando LLM as judge.
        Retorna score (0-1) y explicación.
        """
        # Concatenar contextos
        context_text = "\n\n---\n\n".join(contexts[:3])  # Usar solo los 3 primeros
        
        prompt = f"""Evalúa si la siguiente respuesta es fiel al contexto proporcionado.

CONTEXTO:
{context_text}

PREGUNTA: {question}

RESPUESTA: {answer}

Debes evaluar:
1. ¿La respuesta se basa únicamente en información del contexto?
2. ¿Hay invenciones o información no presente en el contexto?
3. ¿La respuesta es precisa según el contexto?

Responde ÚNICAMENTE con un JSON en este formato exacto:
{{
    "score": <número entre 0 y 1>,
    "explanation": "<breve explicación>"
}}

Donde:
- score=1.0: Totalmente fiel al contexto
- score=0.5: Parcialmente fiel
- score=0.0: Inventado o incorrecto
"""
        
        for attempt in range(max_retries):
            try:
                response = self.llm_judge.invoke(prompt)
                
                # Extraer contenido
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Parsear JSON
                # Buscar el JSON en la respuesta
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    score = float(result.get('score', 0.5))
                    explanation = result.get('explanation', 'Sin explicación')
                    return score, explanation
                else:
                    # Fallback: intentar parsear directamente
                    result = json.loads(response_text)
                    score = float(result.get('score', 0.5))
                    explanation = result.get('explanation', 'Sin explicación')
                    return score, explanation
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    # Último intento fallido
                    return 0.5, f"Error evaluando faithfulness: {e}"
                time.sleep(1)
        
        return 0.5, "No se pudo evaluar"
    
    def evaluate_question(
        self,
        question: str,
        expected_answers: List[str],
        gold_docs: List[str],
        k: int = 10
    ) -> QuestionEvaluation:
        """Evalúa una pregunta completa."""
        
        # 1. Hacer la consulta
        answer, source_docs = self.rag.query(question, use_history=False)
        
        # 2. Extraer nombres de documentos recuperados
        retrieved_doc_names = []
        retrieved_contexts = []
        for doc in source_docs[:k]:
            # Intentar extraer el nombre del documento de los metadatos
            content = doc.page_content
            # El formato es: filename|type|url|...
            first_line = content.split('\n')[0] if '\n' in content else content
            parts = first_line.split('|')
            if parts:
                doc_name = parts[0]
                retrieved_doc_names.append(doc_name)
            else:
                retrieved_doc_names.append("unknown")
            
            retrieved_contexts.append(content)
        
        # DEBUG: Mostrar comparación
        self.console.print(f"\n[yellow]  📋 Gold docs esperados:[/yellow]")
        for i, doc in enumerate(gold_docs, 1):
            self.console.print(f"     {i}. {doc}")
        
        self.console.print(f"\n[cyan]  📄 Docs recuperados (top-10):[/cyan]")
        for i, doc in enumerate(retrieved_doc_names[:10], 1):
            self.console.print(f"     {i}. {doc}")
        
        # 3. Calcular métricas de recuperación
        retrieval_metrics = self.calculate_retrieval_metrics(
            question=question,
            gold_docs=gold_docs,
            retrieved_docs=retrieved_doc_names
        )
        
        # 4. Evaluar faithfulness
        faithfulness_score, faithfulness_explanation = self.evaluate_faithfulness(
            question=question,
            answer=answer,
            contexts=retrieved_contexts
        )
        
        generation_metrics = GenerationMetrics(
            question=question,
            generated_answer=answer,
            expected_answers=expected_answers,
            faithfulness_score=faithfulness_score,
            faithfulness_explanation=faithfulness_explanation
        )
        
        return QuestionEvaluation(
            question=question,
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            retrieved_contexts=retrieved_contexts
        )
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        k: int = 10,
        save_results: bool = True,
        output_dir: str = "validation/results"
    ) -> Dict[str, Any]:
        """Evalúa todo el dataset."""
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=False
        ) as progress:
            task = progress.add_task(
                f"Evaluando {len(dataset)} preguntas...",
                total=len(dataset)
            )
            
            for idx, item in enumerate(dataset, 1):
                question = item['question']
                expected_answers = item.get('answers', [])
                gold_docs = item.get('gold_docs', [])
                
                self.console.print(f"\n[cyan]Pregunta {idx}/{len(dataset)}:[/cyan] {question}")
                
                evaluation = self.evaluate_question(
                    question=question,
                    expected_answers=expected_answers,
                    gold_docs=gold_docs,
                    k=k
                )
                
                results.append(evaluation)
                
                # Mostrar métricas
                self.console.print(f"  Recall: {evaluation.retrieval.recall:.2f}")
                self.console.print(f"  Precision: {evaluation.retrieval.precision:.2f}")
                self.console.print(f"  MRR: {evaluation.retrieval.mrr:.2f}")
                self.console.print(f"  Faithfulness: {evaluation.generation.faithfulness_score:.2f}")
                
                progress.update(task, advance=1)
        
        # Calcular agregados
        aggregated_metrics = self._aggregate_metrics(results)
        
        # Guardar resultados si es necesario
        if save_results:
            self._save_results(results, aggregated_metrics, output_dir)
        
        # Mostrar tabla resumen
        self._print_summary_table(aggregated_metrics)
        
        return {
            'individual_results': results,
            'aggregated_metrics': aggregated_metrics
        }
    
    def _aggregate_metrics(self, results: List[QuestionEvaluation]) -> Dict[str, float]:
        """Agrega métricas de todas las preguntas."""
        
        total = len(results)
        
        avg_recall = sum(r.retrieval.recall for r in results) / total
        avg_precision = sum(r.retrieval.precision for r in results) / total
        avg_mrr = sum(r.retrieval.mrr for r in results) / total
        avg_faithfulness = sum(r.generation.faithfulness_score for r in results) / total
        
        # Calcular F1
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        # Recall@k por niveles
        recalls_at_k = {
            'recall@1': sum(1 for r in results if r.retrieval.rank_first_relevant == 1) / total,
            'recall@5': sum(1 for r in results if 0 < r.retrieval.rank_first_relevant <= 5) / total,
            'recall@10': sum(1 for r in results if 0 < r.retrieval.rank_first_relevant <= 10) / total,
        }
        
        return {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_mrr': avg_mrr,
            'avg_faithfulness': avg_faithfulness,
            'f1_score': f1_score,
            **recalls_at_k,
            'total_questions': total
        }
    
    def _save_results(
        self,
        results: List[QuestionEvaluation],
        aggregated: Dict[str, float],
        output_dir: str
    ):
        """Guarda resultados en archivos."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar resultados individuales en JSON
        individual_path = output_path / "individual_results.json"
        with open(individual_path, 'w', encoding='utf-8') as f:
            json_results = []
            for r in results:
                json_results.append({
                    'question': r.question,
                    'retrieval': asdict(r.retrieval),
                    'generation': asdict(r.generation)
                })
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # Guardar métricas agregadas
        metrics_path = output_path / "aggregated_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        # Crear CSV para análisis
        csv_data = []
        for r in results:
            csv_data.append({
                'question': r.question,
                'recall': r.retrieval.recall,
                'precision': r.retrieval.precision,
                'mrr': r.retrieval.mrr,
                'rank_first_relevant': r.retrieval.rank_first_relevant,
                'faithfulness': r.generation.faithfulness_score,
                'answer': r.generation.generated_answer[:100] + '...'
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        self.console.print(f"\n[green]✓ Resultados guardados en:[/green]")
        self.console.print(f"  - JSON individual: {individual_path}")
        self.console.print(f"  - Métricas agregadas: {metrics_path}")
        self.console.print(f"  - CSV: {csv_path}")
    
    def _print_summary_table(self, metrics: Dict[str, float]):
        """Imprime tabla resumen de métricas."""
        
        table = Table(title="📊 MÉTRICAS DE EVALUACIÓN DEL SISTEMA RAG", show_header=True)
        table.add_column("Métrica", style="cyan", width=30)
        table.add_column("Valor", style="magenta", justify="right", width=15)
        table.add_column("Descripción", style="dim", width=40)
        
        # Métricas de recuperación
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
        
        # Recall@k
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
        
        # Métricas de generación
        table.add_row("", "", "", style="dim")
        table.add_row(
            "Faithfulness promedio",
            f"{metrics['avg_faithfulness']:.3f}",
            "Fidelidad al contexto (LLM judge)"
        )
        
        table.add_row("", "", "", style="dim")
        table.add_row(
            "Total de preguntas",
            f"{metrics['total_questions']}",
            ""
        )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Carga el dataset de evaluación desde JSON."""
    path = Path(dataset_file)
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_file}")


def main():
    import argparse
    from dotenv import load_dotenv
    
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
    # Sistema RAG
    parser.add_argument('--vector_store', type=str, default='validation/validation_vectorstore')  # Usar vectorstore principal
    parser.add_argument('--k', type=int, default=10, help='Documentos a recuperar')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--tfidf_mode', type=str, default='tfidf')
    parser.add_argument('--tfidf_weight', type=float, default=0.3)
    
    # LLM
    parser.add_argument('--provider', type=str, default='claude')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022')
    parser.add_argument('--language', type=str, default='galician')
    
    # Output
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='validation/results',
        help='Directorio para guardar resultados'
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Cargar dataset
    console.print(f"\n[cyan]Cargando dataset de evaluación...[/cyan]")
    dataset = load_dataset(args.dataset)
    console.print(f"[green]✓ {len(dataset)} preguntas cargadas[/green]")
    
    # Inicializar LLMs (query optimizer + answer generator)
    console.print(f"\n[cyan]Inicializando LLMs ({args.provider})...[/cyan]")
    
    # Modelo pequeño para optimizar queries (Haiku)
    llm_query_manager = LLMManager(
        provider=args.provider, 
        model_name='claude-3-5-haiku-20241022'
    )
    llm_query = llm_query_manager.llm
    
    # Modelo principal para respuestas (el que especifiques o Sonnet por defecto)
    llm_manager = LLMManager(provider=args.provider, model_name=args.model)
    llm = llm_manager.llm
    
    console.print(f"[green]✓ LLMs inicializados (Query: Haiku, Answer: {args.model})[/green]")


    # Embeddings y TF-IDF autocontenidos
    from LocalEmbeddings import LocalEmbeddings
    from langchain_community.vectorstores import FAISS
    import pickle
    from DocumentProcessor import DocumentProcessor

    tfidf_dir = Path(args.vector_store)
    text_folder = Path(args.text_folder)
    embeddings = LocalEmbeddings()

    # Si no existe el vectorstore o los archivos de TF-IDF, construye todo
    need_build = not (tfidf_dir.exists() and (tfidf_dir / "index.faiss").exists() and (tfidf_dir / "tfidf_vectorizer.pkl").exists() and (tfidf_dir / "tfidf_matrix.pkl").exists() and (tfidf_dir / "tfidf_documents.pkl").exists())

    if need_build:
        console.print(f"\n[yellow]No se encontró vectorstore o TF-IDF. Construyendo embeddings y TF-IDF...[/yellow]")
        processor = DocumentProcessor(
            docs_folder='validation/crawled_validation',  # Usar datos de validación
            embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
            chunk_size=500,
            chunk_overlap=100,
            verbose=True,
            cache_dir='validation/.doc_cache',
            prefix_mode='source',
            llm=llm,
            map_json='validation/validation_state/map.json',
            crawler_metadata_path='validation/validation_state/metadata.json',
            text_folder=text_folder
        )
        processor.process(force_reload=True, incremental=False)
        processor.save_vectorstore(str(tfidf_dir))
        # Guardar TF-IDF
        tfidf_dir.mkdir(parents=True, exist_ok=True)
        with open(tfidf_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(processor.tfidf_vectorizer, f)
        with open(tfidf_dir / "tfidf_matrix.pkl", "wb") as f:
            pickle.dump(processor.tfidf_matrix, f)
        with open(tfidf_dir / "tfidf_documents.pkl", "wb") as f:
            pickle.dump(processor.tfidf_documents, f)
        console.print(f"[green]✓ Embeddings y TF-IDF construidos y guardados[/green]")

    # Cargar vectorstore
    console.print(f"\n[cyan]Cargando vectorstore...[/cyan]")
    vectorstore = FAISS.load_local(
        str(tfidf_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    console.print(f"[green]✓ Vectorstore cargado[/green]")

    # Cargar componentes TF-IDF
    with open(tfidf_dir / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(tfidf_dir / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(tfidf_dir / "tfidf_documents.pkl", "rb") as f:
        tfidf_documents = pickle.load(f)

    # Inicializar RAG
    console.print(f"\n[cyan]Inicializando sistema RAG...[/cyan]")
    rag = RAGSystem(
        vectorstore=vectorstore,
        k=args.k,
        threshold=args.threshold,
        search_type='mmr',
        language=args.language,
        llm=llm,
        llm_query=llm_query,  # Modelo pequeño para queries
        provider=args.provider,
        temperature=0.1,
        max_tokens=512,
        max_history_length=10,
        use_tfidf=True,
        tfidf_mode=args.tfidf_mode,
        tfidf_weight=args.tfidf_weight,
        tfidf_threshold=0.1,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        tfidf_documents=tfidf_documents
    )
    console.print(f"[green]✓ RAG inicializado[/green]")
    
    # Crear evaluador
    evaluator = RAGEvaluator(
        rag_system=rag,
        llm_judge=llm,
        console=console
    )
    
    # Evaluar
    console.print(f"\n[bold green]Iniciando evaluación...[/bold green]\n")
    
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        k=args.k,
        save_results=True,
        output_dir=args.output_dir
    )
    
    console.print(f"\n[bold green]✓ Evaluación completada[/bold green]")


if __name__ == "__main__":
    main()
