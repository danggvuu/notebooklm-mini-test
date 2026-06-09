"""
Learning Module — Tầng Tạo sinh & Kiểm duyệt
Summarize (Map-Reduce), Quiz, Flashcards generation.
Uses the new retrieval pipeline (Router → HybridSearch/Scroll → Reranker → ContextBuilder).
Pydantic Validation ép kiểu JSON bắt lỗi thiếu đáp án.
"""
import json
from typing import List, Dict, Set, Any, Tuple
from pydantic import ValidationError
from src.config import settings
from src.schemas import Summary, QuizItem, QuizSet, Flashcard, FlashcardSet
from src.rag import retrieve, fetch_all_chunks, render_prompt
from src.retrieval.context_builder import format_citations
from src.llm import invoke_llm

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


def _resolve_target(document, query, filters, k, retrieval_k) -> Tuple[List[Any], str, str]:
    """
    Resolve retrieval target using Scope Router logic:
    - query → HybridSearch (retrieve)
    - no query → Scroll All Chunks (fetch_all_chunks)
    """
    effective_filters = dict(filters or {})

    if document:
        effective_filters["filename"] = document

    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        return chunks, "query", query

    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{k}={v}" for k, v in effective_filters.items())
        return chunks, scope, target

    return fetch_all_chunks(filters=None), "corpus", None


def _parse_json(text: str) -> Any:
    """Parse JSON from LLM output, handling markdown code blocks."""
    cleaned = text.strip()

    # Clean standard markdown blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("\n", 1)[-1].removesuffix("```").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].removesuffix("```").strip()
    elif cleaned.startswith("'''"):
        cleaned = cleaned.split("\n", 1)[-1].removesuffix("'''").strip()

    # Extra safety extraction
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[-1].split("```")[0].strip()

    obj = json.loads(cleaned)
    if not isinstance(obj, (dict, list)):
        raise RuntimeError("Expected JSON object or array.")
    return obj


def _validate_summary_payload(payload: Any) -> Tuple[str, List[str]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Expected dict payload for summary.")
    summary_text = payload.get("summary", "")
    key_points = payload.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = [key_points] if key_points else []
    return str(summary_text), [str(kp) for kp in key_points]


def summarize(document=None, query=None, filters=None, k=None) -> Summary:
    """
    Generate summary using Map-Reduce strategy.
    Uses Qdrant_Scroll for full-document operations (gom lô dữ liệu).
    """
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.summarize_retrieval_k
    )

    if not chunks:
        return Summary(
            scope=scope,
            target=target,
            summary="Không có tài liệu nào để tóm tắt.",
            key_points=[],
            citations=[],
            chunks=[]
        )

    if len(chunks) <= settings.summarize_batch_size:
        prompt = render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)
        payload = _parse_json(invoke_llm(prompt))
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        # Map-Reduce: nhóm dữ liệu tóm tắt văn bản dài
        partials = []
        for start in range(0, len(chunks), settings.summarize_batch_size):
            batch = chunks[start : start + settings.summarize_batch_size]
            payload = _parse_json(invoke_llm(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)))
            summary_text, key_points = _validate_summary_payload(payload)
            partials.append({"summary": summary_text, "key_points": key_points})

        payload = _parse_json(invoke_llm(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials)))
        summary_text, key_points = _validate_summary_payload(payload)

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
        chunks=chunks,
    )


def _validate_items(payload, key, model_class, dedup_field, label, valid_markers):
    """
    Pydantic Validation: ép kiểu JSON bắt lỗi thiếu đáp án.
    Validates and deduplicates generated items.
    """
    raw_items = payload.get(key)
    if not raw_items:
        raw_items = []

    items, seen = [], set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue

        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue

        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))

    if not items:
        raise RuntimeError(f"No valid {label} produced.")
    return items


def generate_quiz(document=None, query=None, filters=None, count=None, k=None) -> QuizSet:
    """Generate multiple-choice quiz with Pydantic validation."""
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )

    if not chunks:
        return QuizSet(scope=scope, target=target, items=[], citations=[], chunks=[])

    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))

    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers)

    return QuizSet(
        scope=scope,
        target=target,
        items=items,
        chunks=chunks,
        citations=format_citations(chunks)
    )


def generate_flashcards(document=None, query=None, filters=None, count=None, k=None) -> FlashcardSet:
    """Generate flashcards with Pydantic validation."""
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )

    if not chunks:
        return FlashcardSet(scope=scope, target=target, cards=[], citations=[], chunks=[])

    n = count or settings.flashcards_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))

    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)

    return FlashcardSet(
        scope=scope,
        target=target,
        cards=cards,
        chunks=chunks,
        citations=format_citations(chunks)
    )
