import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.config import settings
from src.schemas import RagAnswer
from src.indexing import ingest
from src.rag import answer
from src.store import get_embeddings
from src.evaluation.chunking_strategies import (
    ChunkingStrategy, 
    RecursiveChunker, 
    SemanticChunkerWrapper, 
    _RECURSIVE_CONFIGS, 
    _SEMANTIC_CONFIGS
)
from src.evaluation.ragas_evaluator import run_evaluation

def summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    res = {}
    for m in metrics:
        if m in df.columns:
            res[m] = float(df[m].mean())
    return res

def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _evaluate_strategy(
    strategy: ChunkingStrategy, 
    output_dir: Path, 
    test_cases: List[Dict[str, str]]
) -> Dict[str, Any]:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    chunk_count = ingest(
        recreate=True, 
        collection_name=collection_name, 
        chunker=strategy.chunker
    )
    
    result_out: Dict[str, Any] = {
        "strategy_id": strategy.strategy_id,
        "chunk_count": chunk_count,
        "summary_metrics": {},
    }
    
    try:
        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)
            
        result = run_evaluation(
            test_cases, 
            answer_fn=answer_fn, 
            llm_provider=settings.llm_provider
        )
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)
        
    except Exception as exc:
        result_out["error"] = str(exc)
        
    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    return result_out

def load_test_cases(csv_path: Path) -> List[Dict[str, str]]:
    cases = []
    if not csv_path.exists():
        return cases
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"]
            })
    return cases

def main():
    csv_path = Path(__file__).parent / "benchmark_rag.csv"
    test_cases = load_test_cases(csv_path)
    if not test_cases:
        print(f"No test cases found in {csv_path}. Please populate the benchmark.")
        return
        
    output_dir = Path("storage/evaluation/chunking")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strategies = []
    
    # 1. Load Recursive Strategies
    for sid, size, overlap in _RECURSIVE_CONFIGS:
        chunker = RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
        strategies.append(
            ChunkingStrategy(
                strategy_id=sid, 
                chunker=chunker, 
                params={"size": size, "overlap": overlap}
            )
        )
        
    # 2. Load Semantic Strategies
    embeddings = get_embeddings()
    for sid, btype in _SEMANTIC_CONFIGS:
        chunker = SemanticChunkerWrapper(embeddings=embeddings, breakpoint_type=btype)
        strategies.append(
            ChunkingStrategy(
                strategy_id=sid, 
                chunker=chunker, 
                params={"breakpoint_type": btype}
            )
        )
        
    print(f"Loaded {len(strategies)} chunking strategies for evaluation.")
    
    results = []
    for s in strategies:
        print(f"\nEvaluating strategy: {s.strategy_id}...")
        res = _evaluate_strategy(s, output_dir, test_cases)
        results.append(res)
        print(f"Strategy {s.strategy_id} completed. Metrics: {res.get('summary_metrics', {})}")
        
    # Print summary table
    print("\n" + "="*50)
    print("CHUNKING EVALUATION SUMMARY")
    print("="*50)
    print(f"{'Strategy ID':<25} | {'Chunks':<6} | {'Faith.':<6} | {'Relev.':<6} | {'Prec.':<6} | {'Recall':<6}")
    print("-"*50)
    for r in results:
        m = r.get("summary_metrics", {})
        err = r.get("error")
        if err:
            print(f"{r['strategy_id']:<25} | Error: {err[:35]}...")
        else:
            print(f"{r['strategy_id']:<25} | {r['chunk_count']:<6} | {m.get('faithfulness', 0.0):.4f} | {m.get('answer_relevancy', 0.0):.4f} | {m.get('context_precision', 0.0):.4f} | {m.get('context_recall', 0.0):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
