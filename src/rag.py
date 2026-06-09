"""
RAG Orchestrator — Main pipeline
Router → Cache Check → HybridSearch → Reranker → Context Builder → LLM
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Iterator, List

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from src.config import settings
from src.schemas import ChunkMetadata, RetrievedChunk, Citation, RagAnswer
from src.store import get_vector_store, get_client
from src.filters import filters_to_qdrant
from src.llm import invoke_llm, stream_llm
from src.cache import get_cache
from src.observability import record_cache_hit, record_cache_miss
from src.retrieval.router import get_router
from src.retrieval.hybrid_search import get_hybrid_searcher
from src.retrieval.reranker import get_reranker
from src.retrieval.context_builder import get_context_builder, format_citations

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"


# ---------------------------------------------------------------------------
# Retrieval functions (backward compatible)
# ---------------------------------------------------------------------------

def retrieve(query, k=None, filters=None, collection_name=None):
    """
    Backward-compatible retrieval using Hybrid Search + Reranker.
    """
    # Use hybrid search
    searcher = get_hybrid_searcher()
    initial_k = k or settings.hybrid_initial_k
    chunks = searcher.search(
        query=query,
        k=initial_k,
        filters=filters,
        collection_name=collection_name,
    )

    # Rerank
    reranker = get_reranker()
    rerank_k = settings.hybrid_rerank_k
    reranked = reranker.rerank(query, chunks, top_k=rerank_k)

    return reranked


def scroll_all(collection_name, scroll_filter=None):
    """Scroll all chunks from Qdrant (no search algorithm)."""
    client = get_client()
    offset = None
    while True:
        res, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not res:
            break
        yield res
        if next_offset is None:
            break
        offset = next_offset


def fetch_all_chunks(filters=None, collection_name=None):
    """Fetch all chunks via Qdrant scroll (for full-document operations)."""
    name = collection_name or settings.qdrant_collection
    results = []

    if not get_client().collection_exists(name):
        return []

    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""
            if meta and text:
                results.append(RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta)))

    return sorted(results, key=lambda r: (
        r.metadata.filename,
        r.metadata.page,
        int(r.metadata.chunk_id.rsplit(":", 1)[-1]),
    ))


# ---------------------------------------------------------------------------
# Jinja2 Prompt Rendering
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _jinja_env():
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name, **context):
    return _jinja_env().get_template(template_name).render(**context)


# ---------------------------------------------------------------------------
# Main Answer Pipeline
# ---------------------------------------------------------------------------

def answer(question, k=None, filters=None, collection_name=None, session_id=None) -> RagAnswer:
    """
    Main RAG pipeline:
    1. Check Redis Semantic Cache
    2. Route via ScopeRouter
    3. HybridSearch (Semantic + BM25)
    4. Cross-Encoder Reranker
    5. Context Builder → Jinja2 Prompt
    6. LLM invoke
    7. Cache result
    """
    # Step 1: Check cache
    cache = get_cache()
    cached = cache.get(question)
    if cached:
        record_cache_hit()
        logger.info("Cache HIT — returning cached answer.")
        return RagAnswer(**cached)

    record_cache_miss()

    # Step 2-4: Retrieve + Rerank
    router = get_router()
    scope = router.resolve(query=question, filters=filters, operation="ask")

    if scope.scope_type == "query":
        chunks = retrieve(question, k=k, filters=scope.filters, collection_name=collection_name)
    else:
        chunks = fetch_all_chunks(filters=scope.filters, collection_name=collection_name)

    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )

    # Step 5: Build context & render prompt
    context = get_context_builder().build(chunks, scope="query", target=question)
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=context.chunks)

    # Step 6: Invoke LLM
    text = invoke_llm(prompt)

    result = RagAnswer(
        question=question,
        answer=text.strip(),
        citations=context.citations,
        chunks=context.chunks,
    )

    # Step 7: Cache the result
    cache.put(question, result.model_dump())

    return result


def answer_stream(question, k=None, filters=None, collection_name=None) -> Iterator[str]:
    """
    Streaming RAG pipeline.
    Returns an iterator of text chunks for SSE streaming.
    Uses StreamBatcher to buffer tokens.
    """
    from src.stream_batching import get_stream_batcher

    # Check cache first
    cache = get_cache()
    cached = cache.get(question)
    if cached:
        record_cache_hit()
        # Return cached answer as single chunk
        yield cached.get("answer", "")
        return

    record_cache_miss()

    # Retrieve + Rerank
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)

    if not chunks:
        yield "Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        return

    # Build context & render prompt
    context = get_context_builder().build(chunks, scope="query", target=question)
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=context.chunks)

    # Stream LLM response through batcher
    batcher = get_stream_batcher()
    token_stream = stream_llm(prompt)

    collected = []
    for batch in batcher.batch(token_stream):
        collected.append(batch)
        yield batch

    # Cache the full answer
    full_answer = "".join(collected)
    result = RagAnswer(
        question=question,
        answer=full_answer.strip(),
        citations=context.citations,
        chunks=context.chunks,
    )
    cache.put(question, result.model_dump())
