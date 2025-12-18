"""
Shared utilities for RAG validation and grid search.
Contains common functions, classes, and evaluation logic.
"""
import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

from langchain_community.vectorstores import FAISS

from src.rag.RAGSystem import RAGSystem
from src.LLMManager import LLMManager
from src.LocalEmbeddings import LocalEmbeddings
from src.preprocessing.DocumentProcessor import DocumentProcessor


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
    mrr: float
    rank_first_relevant: int


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
        llm_judge: Any,
        console: Any = None,
        verbose: bool = True
    ):
        self.rag = rag_system
        self.llm_judge = llm_judge
        self.console = console
        self.verbose = verbose
    
    def calculate_retrieval_metrics(
        self,
        question: str,
        gold_docs: List[str],
        retrieved_docs: List[str]
    ) -> RetrievalMetrics:
        """Calcula métricas de recuperación."""
        
        def normalize_doc_name(doc: str) -> str:
            name = Path(doc).name.lower()
            while name.endswith('.txt'):
                name = name[:-4]
            return name
        
        gold_normalized = {normalize_doc_name(d) for d in gold_docs}
        retrieved_normalized = [normalize_doc_name(d) for d in retrieved_docs]
        
        retrieved_unique = set(retrieved_normalized)
        relevant_docs_found = len(gold_normalized.intersection(retrieved_unique))
        recall = relevant_docs_found / len(gold_normalized) if gold_normalized else 0.0
        
        relevant_chunks = sum(1 for d in retrieved_normalized if d in gold_normalized)
        precision = relevant_chunks / len(retrieved_normalized) if retrieved_normalized else 0.0
        
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
        """Evalúa la fidelidad de la respuesta usando LLM as judge."""
        
        def clean_context(text: str) -> str:
            return " ".join(text.split())

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
                response_text = response.content if hasattr(response, 'content') else str(response)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                
                if json_match:
                    result = json.loads(json_match.group())
                    score = float(result.get('score', 0.0))
                    claims = result.get('claims_count', 0)
                    supported = result.get('supported_claims', 0)
                    explanation = result.get('explanation', 'Sin explicación')
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
        """Evalúa si la respuesta es pertinente."""
        
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
        
        answer, source_docs = self.rag.query(question, use_history=False)
        
        retrieved_doc_names = []
        retrieved_contexts = []
        for doc in source_docs[:k]:
            content = doc.page_content
            first_line = content.split('\n')[0] if '\n' in content else content
            parts = first_line.split('|')
            if parts:
                doc_name = parts[0]
                retrieved_doc_names.append(doc_name)
            else:
                retrieved_doc_names.append("unknown")
            
            retrieved_contexts.append(content)
        
        if self.verbose and self.console:
            self.console.print(f"\n[yellow]  📋 Gold docs esperados:[/yellow]")
            for i, doc in enumerate(gold_docs, 1):
                self.console.print(f"     {i}. {doc}")
            
            self.console.print(f"\n[cyan]  📄 Docs recuperados (top-10):[/cyan]")
            for i, doc in enumerate(retrieved_doc_names[:10], 1):
                self.console.print(f"     {i}. {doc}")
        
        retrieval_metrics = self.calculate_retrieval_metrics(
            question=question,
            gold_docs=gold_docs,
            retrieved_docs=retrieved_doc_names
        )
        
        faithfulness_score, faithfulness_explanation = self.evaluate_faithfulness(
            question=question,
            answer=answer,
            contexts=retrieved_contexts
        )

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
        config_name: str = "",
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Evalúa todo el dataset."""
        from tqdm import tqdm
        
        results = []
        
        iterator = tqdm(dataset, desc="Evaluating questions", disable=not show_progress) if show_progress else dataset
        
        for idx, item in enumerate(iterator, 1):
            question = item['question']
            expected_answers = item.get('answers', [])
            gold_docs = item.get('gold_docs', [])
            
            if self.verbose and self.console:
                self.console.print(f"\n[cyan]Pregunta {idx}/{len(dataset)}:[/cyan] {question}")
            
            evaluation = self.evaluate_question(
                question=question,
                expected_answers=expected_answers,
                gold_docs=gold_docs,
                k=k
            )
            
            results.append(evaluation)
            
            if self.verbose and self.console:
                self.console.print(f"  Recall: {evaluation.retrieval.recall:.2f}")
                self.console.print(f"  Precision: {evaluation.retrieval.precision:.2f}")
                self.console.print(f"  MRR: {evaluation.retrieval.mrr:.2f}")
                self.console.print(f"  Faithfulness: {evaluation.generation.faithfulness_score:.2f}")
                self.console.print(f"    Explicación: {evaluation.generation.faithfulness_explanation}")
                self.console.print(f"  Relevance: {evaluation.generation.answer_relevance_score:.2f}")
                self.console.print(f"    Explicación: {evaluation.generation.answer_relevance_explanation}")
        
        aggregated_metrics = self._aggregate_metrics(results)
        
        if save_results:
            self._save_results(results, aggregated_metrics, output_dir, config_name)
        
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
        
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
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
        """Guarda resultados en archivos."""
        import pandas as pd
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        suffix = f"_{config_name}" if config_name else ""
        
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
        
        metrics_path = output_path / f"aggregated_metrics{suffix}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
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
        
        if self.verbose and self.console:
            self.console.print(f"\n[green]✓ Resultados guardados en:[/green]")
            self.console.print(f"  - JSON individual: {individual_path}")
            self.console.print(f"  - Métricas agregadas: {metrics_path}")
            self.console.print(f"  - CSV: {csv_path}")


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Carga el dataset de evaluación desde JSON."""
    path = Path(dataset_file)
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_file}")


def setup_llms(
    provider: str,
    model: str,
    judge_provider: str,
    judge_model: str
) -> Tuple[Any, Any, Any]:
    """
    Inicializa los LLMs necesarios.
    Returns: (llm, llm_query, llm_judge)
    """
    llm_query_manager = LLMManager(
        provider=provider, 
        model_name='claude-3-5-haiku-20241022'
    )
    llm_query = llm_query_manager.llm
    
    llm_manager = LLMManager(provider=provider, model_name=model)
    llm = llm_manager.llm
    
    if judge_provider == 'openai':
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY no encontrada en .env")
        llm_judge = OpenAIJudge(api_key=openai_key, model=judge_model)
    else:
        llm_judge_manager = LLMManager(provider=judge_provider, model_name=judge_model)
        llm_judge = llm_judge_manager.llm
    
    return llm, llm_query, llm_judge


def setup_rag_system(
    vectorstore_path: str,
    text_folder: str,
    state_dir: str,
    chunk_size: int,
    bm25_weight: float,
    bm25_mode: str,
    k: int,
    threshold: float,
    language: str,
    llm: Any,
    llm_query: Any,
    provider: str,
    use_bm25: bool = True,
    verbose: bool = False
) -> Optional[RAGSystem]:
    """Configura el sistema RAG con los parámetros dados."""
    
    chunk_overlap = int(chunk_size * 0.2)
    vectorstore_dir = Path(vectorstore_path)
    embeddings = LocalEmbeddings()
    
    need_build = not (
        vectorstore_dir.exists() and 
        (vectorstore_dir / "index.faiss").exists() and 
        (vectorstore_dir / "bm25_index.pkl").exists()
    )
    
    if need_build:
        if verbose:
            print(f"[BUILD] Construyendo vectorstore para chunk_size={chunk_size}...")
        
        processor = DocumentProcessor(
            docs_folder='validation/crawled_validation',
            embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose,
            cache_dir='validation/.doc_cache',
            prefix_mode='source',
            llm=llm,
            crawler_metadata_path='validation/validation_state/metadata.json',
            text_folder=text_folder
        )
        processor.process(force_reload=True, incremental=False)
        processor.save_vectorstore(str(vectorstore_dir))
        
        if verbose:
            print(f"[BUILD] ✓ Vectorstore construido")
    
    vectorstore = FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    processor = DocumentProcessor(
        docs_folder='validation/crawled_validation',
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=False,
        cache_dir='validation/.doc_cache',
        prefix_mode='source',
        crawler_metadata_path='validation/validation_state/metadata.json',
        text_folder=text_folder
    )
    processor.load_vectorstore(str(vectorstore_dir))
    
    rag_params = {
        'vectorstore': vectorstore,
        'k': k,
        'threshold': threshold,
        'state_dir': str(state_dir),
        'search_type': 'mmr',
        'language': language,
        'llm': llm,
        'llm_query': llm_query,
        'provider': provider,
        'temperature': 0.1,
        'max_tokens': 512,
        'max_history_length': 10,
        'use_bm25': use_bm25,
        'bm25_mode': bm25_mode,
        'bm25_weight': bm25_weight,
        'bm25_threshold': 0.1
    }
    
    if use_bm25 and (bm25_mode == "hybrid" or bm25_mode == "bm25"):
        if processor.bm25_index is None or processor.bm25_documents is None:
            if verbose:
                print(f"[WARN] BM25 components not found, disabling BM25")
            rag_params['use_bm25'] = False
        else:
            rag_params.update({
                'bm25_index': processor.bm25_index,
                'bm25_documents': processor.bm25_documents
            })
    
    return RAGSystem(**rag_params)
