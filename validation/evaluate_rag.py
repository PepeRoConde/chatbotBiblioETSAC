"""
Script de evaluación del sistema RAG.
Métricas: Recall, MRR, Precision, Faithfulness (LLM as judge)
"""
import sys
import json
import time
import os
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

from RAGSystem.RAGSystem import RAGSystem
from LLMManager import LLMManager


class OpenAIJudge:
    """Cliente directo de OpenAI para evaluación."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"OpenAI Judge initialized with model: {model}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    def invoke(self, prompt: str) -> Any:
        """Simula la interfaz de LangChain para compatibilidad."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )
        
        # Crear objeto compatible con LangChain
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(response.choices[0].message.content)


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
    faithfulness_score: float
    faithfulness_explanation: str
    answer_relevance_score: float
    answer_relevance_explanation: str 


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
            max_retries: int = 3
        ) -> Tuple[float, str]:
            """
            Evalúa la fidelidad de la respuesta usando LLM as judge con lógica de NLI.
            Retorna score (0-1) y explicación detallada.
            """
            # Función interna para limpiar cada chunk y ahorrar tokens
            def clean_context(text: str) -> str:
                return " ".join(text.split())

            # Unimos TODO el contexto que el generador tuvo disponible, limpiando cada parte
            context_text = "\n\n---\n\n".join([clean_context(c) for c in contexts])
            
            prompt = f"""Instrucciones: Evalúa la fidelidad (Faithfulness) de la respuesta basándote EXCLUSIVAMENTE en el contexto proporcionado.
            
            CONTEXTO:
            {context_text}
            
            PREGUNTA: {question}
            
            RESPUESTA A EVALUAR:
            {answer}
            
            Tarea de evaluación:
            1. Divide la respuesta proporcionada en afirmaciones (claims) individuales y atómicas.
            2. Para cada afirmación, verifica si el contexto la respalda directamente.
            3. El score final es: (afirmaciones respaldadas / total de afirmaciones).
            
            Reglas estrictas:
            - Si la respuesta contiene información que es verdadera en el mundo real pero NO aparece en el contexto, esa afirmación NO está respaldada.
            - Si la respuesta dice "No lo sé" o similar porque la información no está en el contexto, y efectivamente no está, el score debe ser 1.0.
            - Debes entender tanto el gallego como el español perfectamente y no penalizar respuestas en uno u otro idioma.
            
            Responde ÚNICAMENTE con un JSON en este formato exacto:
            {{
                "score": <0.0 a 1.0>,
                "claims_count": <int>,
                "supported_claims": <int>,
                "explanation": "<explicación breve de qué afirmaciones fallaron y por qué>"
            }}
            """
            
            for attempt in range(max_retries):
                try:
                    response = self.llm_judge.invoke(prompt)
                    
                    # Extraer contenido del objeto de respuesta
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Buscar el bloque JSON mediante regex
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    
                    if json_match:
                        result = json.loads(json_match.group())
                        
                        # Extraemos los datos del JSON
                        score = float(result.get('score', 0.0))
                        claims = result.get('claims_count', 0)
                        supported = result.get('supported_claims', 0)
                        explanation = result.get('explanation', 'Sin explicación')
                        
                        # Formateamos una explicación técnica que incluye el conteo de claims
                        full_explanation = f"[{supported}/{claims} claims] {explanation}"
                        
                        return score, full_explanation
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        return 0.0, f"Error crítico en evaluación: {str(e)}"
                    time.sleep(1)
                    
            return 0.0, "Fallo en el parseo del juicio del LLM tras varios intentos"
    
    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
        max_retries: int = 3
    ) -> Tuple[float, str]:
        """
        Evalúa si la respuesta es pertinente y resuelve la duda planteada.
        No juzga la veracidad (eso es Faithfulness), sino la relevancia.
        """
        
        prompt = f"""Instrucciones: Evalúa la relevancia de la respuesta respecto a la pregunta planteada.
        Debes entender tanto el gallego como el español perfectamente y no penalizar respuestas en uno u otro idioma.
        
        PREGUNTA: {question}
        
        RESPUESTA A EVALUAR:
        {answer}
        
        Tarea de evaluación:
        1. ¿La respuesta aborda directamente todos los puntos de la pregunta?
        2. ¿La respuesta es concisa y evita información irrelevante que no se pidió?
        3. ¿El tono y el formato son adecuados para la consulta?

        Criterios de Score:
        - 1.0: La respuesta resuelve perfectamente la duda de forma directa.
        - 0.7: Resuelve la duda pero incluye mucha paja o información no solicitada.
        - 0.3: Aborda el tema pero no responde a la duda específica.
        - 0.0: La respuesta es irrelevante, ignora la pregunta o es un mensaje de error genérico.

        Responde ÚNICAMENTE con un JSON en este formato exacto:
        {{
            "score": <0.0 a 1.0>,
            "explanation": "<explicación breve de por qué la respuesta es o no relevante>"
        }}
        """
        
        for attempt in range(max_retries):
            try:
                response = self.llm_judge.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group())
                    score = float(result.get('score', 0.0))
                    explanation = result.get('explanation', 'Sin explicación')
                    
                    return score, explanation
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return 0.0, f"Error en relevancia: {str(e)}"
                time.sleep(1)
                
        return 0.0, "Fallo en el juicio de relevancia"
    
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

        # 5. Evaluar relevancia
        relevance_score, relevance_explanation = self.evaluate_answer_relevance(
            question=question,
            answer=answer
        )
        
        generation_metrics = GenerationMetrics(
            question=question,
            generated_answer=answer,
            expected_answers=expected_answers,
            faithfulness_score=faithfulness_score,
            faithfulness_explanation=faithfulness_explanation,
            answer_relevance_score=relevance_score,
            answer_relevance_explanation=relevance_explanation
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
        output_dir: str = "validation/results",
        config_name: str = ""
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
                self.console.print(f"    Explicación: {evaluation.generation.faithfulness_explanation}")
                self.console.print(f"  Relevance: {evaluation.generation.answer_relevance_score:.2f}")
                self.console.print(f"    Explicación: {evaluation.generation.answer_relevance_explanation}")
                
                progress.update(task, advance=1)
        
        # Calcular agregados
        aggregated_metrics = self._aggregate_metrics(results)
        
        # Guardar resultados si es necesario
        if save_results:
            self._save_results(results, aggregated_metrics, output_dir, config_name)
        
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
        avg_relevance = sum(r.generation.answer_relevance_score for r in results) / total
        
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
            'avg_relevance': avg_relevance,
            'f1_score': f1_score,
            **recalls_at_k,
            'total_questions': total
        }
    
    def _save_results(
        self,
        results: List[QuestionEvaluation],
        aggregated: Dict[str, float],
        output_dir: str,
        config_name: str = ""
    ):
        """Guarda resultados en archivos con nombre descriptivo."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Suffix para archivos
        suffix = f"_{config_name}" if config_name else ""
        
        # Guardar resultados individuales en JSON
        individual_path = output_path / f"individual_results{suffix}.json"
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
        metrics_path = output_path / f"aggregated_metrics{suffix}.json"
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
                'relevance': r.generation.answer_relevance_score,
                'answer': r.generation.generated_answer[:100] + '...'
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = output_path / f"results{suffix}.csv"
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
    parser.add_argument('--state_dir', type=str, default='validation/validation_state')
    # Sistema RAG
    parser.add_argument('--vector_store', type=str, default='validation/validation_vectorstore')
    parser.add_argument('--k', type=int, default=10, help='Documentos a recuperar')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--tfidf_mode', type=str, default='tfidf', 
                        choices=['tfidf', 'hybrid', 'rerank'], 
                        help='Modo de TF-IDF: tfidf (solo TFIDF), hybrid (FAISS+TFIDF), rerank (FAISS reranked)')
    parser.add_argument('--tfidf_weight', type=float, default=0.3)
    parser.add_argument('--chunk_size', type=int, default=500, help='Tamaño de chunks')
    parser.add_argument('--use_tfidf', action='store_true', default=True, help='Usar TF-IDF')
    parser.add_argument('--no_tfidf', action='store_false', dest='use_tfidf', help='No usar TF-IDF')
    
    # LLM
    parser.add_argument('--provider', type=str, default='claude', help='Provider para generación de respuestas')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022', help='Modelo para generación')
    parser.add_argument('--judge_provider', type=str, default='openai', help='Provider para evaluación (openai o claude)')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini', help='Modelo para evaluación')
    parser.add_argument('--language', type=str, default='galician')
    
    # Output
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='validation/results',
        help='Directorio para guardar resultados'
    )
    
    args = parser.parse_args()
    
    # Calcular overlap automáticamente (10% del chunk_size hacia delante y atrás = 20% total)
    chunk_overlap = int(args.chunk_size * 0.2)
    
    # Crear nombre de configuración para archivos
    config_name = f"{args.tfidf_mode}_chunk{args.chunk_size}"
    
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
    
    # Modelo para evaluación (judge)
    if args.judge_provider == 'openai':
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY no encontrada en .env")
        llm_judge = OpenAIJudge(api_key=openai_key, model=args.judge_model)
    else:
        llm_judge_manager = LLMManager(provider=args.judge_provider, model_name=args.judge_model)
        llm_judge = llm_judge_manager.llm
    
    console.print(f"[green]✓ LLMs inicializados:[/green]")
    console.print(f"  - Query optimization: Haiku")
    console.print(f"  - Answer generation: {args.model}")
    console.print(f"  - Evaluation (judge): {args.judge_provider}/{args.judge_model}")
    console.print(f"[yellow]Configuración:[/yellow]")
    console.print(f"  - Chunk size: {args.chunk_size}")
    console.print(f"  - Chunk overlap: {chunk_overlap} (20% del chunk_size)")
    console.print(f"  - TF-IDF mode: {args.tfidf_mode}")
    console.print(f"  - Use TF-IDF: {args.use_tfidf}")


    # Embeddings y TF-IDF autocontenidos
    from LocalEmbeddings import LocalEmbeddings
    from langchain_community.vectorstores import FAISS
    import pickle
    from DocumentProcessor import DocumentProcessor

    tfidf_dir = Path(args.vector_store)
    text_folder = Path(args.text_folder)
    state_dir = Path(args.state_dir)
    embeddings = LocalEmbeddings()

    # Si no existe el vectorstore o los archivos de TF-IDF, construye todo
    need_build = not (tfidf_dir.exists() and (tfidf_dir / "index.faiss").exists() and (tfidf_dir / "tfidf_vectorizer.pkl").exists() and (tfidf_dir / "tfidf_matrix.pkl").exists() and (tfidf_dir / "tfidf_documents.pkl").exists())

    if need_build:
        console.print(f"\n[yellow]No se encontró vectorstore o TF-IDF. Construyendo embeddings y TF-IDF...[/yellow]")
        processor = DocumentProcessor(
            docs_folder='validation/crawled_validation',  # Usar datos de validación
            embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
            chunk_size=args.chunk_size,
            chunk_overlap=chunk_overlap,
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
        state_dir=state_dir,
        search_type='mmr',
        language=args.language,
        llm=llm,
        llm_query=llm_query,  # Modelo pequeño para queries
        provider=args.provider,
        temperature=0.1,
        max_tokens=512,
        max_history_length=10,
        use_tfidf=args.use_tfidf,
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
        llm_judge=llm_judge,  # Usar GPT u otro modelo para evaluación
        console=console
    )
    
    # Evaluar
    console.print(f"\n[bold green]Iniciando evaluación...[/bold green]\n")
    
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        k=args.k,
        save_results=True,
        output_dir=args.output_dir,
        config_name=config_name
    )
    
    console.print(f"\n[bold green]✓ Evaluación completada[/bold green]")


if __name__ == "__main__":
    main()
