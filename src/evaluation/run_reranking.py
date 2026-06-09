import csv
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sentence_transformers import CrossEncoder
from src.config import settings
from src.schemas import RagAnswer
from src.rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from src.llm import invoke_llm
from src.evaluation.ragas_evaluator import run_evaluation, summary_metrics
from src.evaluation.run_chunking import load_test_cases, write_json

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

def answer_with_reranker(
    question: str,
    collection_name: str,
    reranker: CrossEncoder,
    initial_k: int = 15,
    rerank_k: int = 5,
    filters: dict = None,
) -> RagAnswer:
    # Giai đoạn 1: Truy xuất thô (initial_k)
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )
        
    # Giai đoạn 2: Tính toán điểm số tương quan chéo bằng Cross-Encoder
    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
        
    # Xếp hạng lại và lọc ra các đoạn liên quan nhất (rerank_k)
    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]
    
    # Đưa ngữ cảnh đã được lọc vào prompt cho LLM
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked)
    text = invoke_llm(prompt)
    
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(reranked),
        chunks=reranked,
    )

def main():
    csv_path = Path(__file__).parent / "benchmark_rag.csv"
    test_cases = load_test_cases(csv_path)
    if not test_cases:
        print(f"No test cases found in {csv_path}. Please populate the benchmark.")
        return
        
    output_path = Path("storage/evaluation/reranking/recursive_rerank.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading CrossEncoder model: {RERANKER_MODEL}...")
    try:
        reranker = CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        print(f"Error loading CrossEncoder model locally: {e}. Falling back to mock/CPU load.")
        reranker = None
        
    if not reranker:
        print("Reranking evaluation cannot proceed without a valid CrossEncoder.")
        return
        
    # Run evaluation using the standard Best Chunker collection
    best_collection = f"{settings.qdrant_collection}__rc_1000_150"
    
    print("\nRunning RAG evaluation WITH Cross-Encoder Reranker...")
    
    def answer_fn(q: str) -> RagAnswer:
        return answer_with_reranker(
            question=q,
            collection_name=best_collection,
            reranker=reranker,
            initial_k=15,
            rerank_k=5
        )
        
    result_out = {
        "strategy_id": "recursive_1000_150_with_reranker",
        "summary_metrics": {},
    }
    
    try:
        result = run_evaluation(
            test_cases, 
            answer_fn=answer_fn, 
            llm_provider=settings.llm_provider
        )
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)
        print(f"\nReranking evaluation completed. Metrics: {result_out['summary_metrics']}")
    except Exception as exc:
        result_out["error"] = str(exc)
        print(f"\nEvaluation failed: {exc}")
        
    write_json(output_path, result_out)

if __name__ == "__main__":
    main()
