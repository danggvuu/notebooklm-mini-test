"""
FastAPI Backend — Tầng Giao diện & Định tuyến API
Hỗ trợ Server-Sent Events (SSE), async upload, session management,
feedback, và Prometheus metrics.
"""
import json
import logging

from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import settings
from src.schemas import RagAnswer, Summary, QuizSet, FlashcardSet
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import save_and_ingest_file
from src.rag import answer, answer_stream
from src.learning import summarize as summarize_learning, generate_quiz, generate_flashcards
from src.store import get_client
from src.worker import get_task_tracker, process_file_background, TaskInfo
from src.session import get_session_store
from src.observability import (
    track_latency, record_feedback, get_metrics_response, init_langsmith,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed documents. "
                "Supports SSE streaming, async upload, session management, and observability.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    """Initialize observability on startup."""
    init_langsmith()
    logger.info("RAG API v2.0.0 started.")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: Optional[int] = Field(default=None, ge=1, le=64)
    filters: Optional[MetadataFilter] = None
    session_id: Optional[str] = None

class SummarizeRequest(BaseModel):
    document: Optional[str] = None
    query: Optional[str] = None
    filters: Optional[MetadataFilter] = None
    k: Optional[int] = Field(default=None, ge=1, le=64)

class QuizRequest(BaseModel):
    document: Optional[str] = None
    query: Optional[str] = None
    filters: Optional[MetadataFilter] = None
    count: Optional[int] = Field(default=None, ge=1, le=50)
    k: Optional[int] = Field(default=None, ge=1, le=64)

class FlashcardsRequest(QuizRequest):
    pass

class UploadResponse(BaseModel):
    task_id: str
    filename: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: str
    chunks_indexed: int = 0
    error_message: Optional[str] = None

class FeedbackRequest(BaseModel):
    question: str
    feedback_type: str = Field(pattern="^(up|down)$")
    session_id: Optional[str] = None

class DocumentInfo(BaseModel):
    filename: str
    document_id: str
    num_pages: int
    num_chunks: int


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def list_documents() -> List[DocumentInfo]:
    client = get_client()
    collection_name = settings.qdrant_collection
    if not client.collection_exists(collection_name):
        return []

    offset = None
    docs = {}
    while True:
        res, next_offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in res:
            meta = point.payload.get("metadata") or {}
            doc_id = meta.get("document_id")
            filename = meta.get("filename")
            page = meta.get("page")

            if doc_id and filename:
                if doc_id not in docs:
                    docs[doc_id] = {
                        "filename": filename,
                        "document_id": doc_id,
                        "pages": set(),
                        "chunks_count": 0
                    }
                docs[doc_id]["pages"].add(page)
                docs[doc_id]["chunks_count"] += 1

        if next_offset is None or not res:
            break
        offset = next_offset

    return [
        DocumentInfo(
            filename=info["filename"],
            document_id=doc_id,
            num_pages=len(info["pages"]),
            num_chunks=info["chunks_count"]
        )
        for doc_id, info in docs.items()
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/documents", response_model=List[DocumentInfo])
@track_latency("documents")
def documents():
    return list_documents()


# --- Upload (Async with Background Worker) ---

@app.post("/upload", response_model=UploadResponse)
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a file and process it asynchronously.
    Returns immediately with a task_id for status polling.
    """
    content = await file.read()
    filename = file.filename or "unknown"

    tracker = get_task_tracker()
    task = tracker.create(filename)

    # Schedule background processing
    background_tasks.add_task(process_file_background, content, filename, task)

    return UploadResponse(
        task_id=task.task_id,
        filename=filename,
        status="pending",
    )


@app.get("/upload/status/{task_id}", response_model=TaskStatusResponse)
def upload_status(task_id: str):
    """Check the status of a background upload task."""
    tracker = get_task_tracker()
    task = tracker.get(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"detail": f"Task {task_id} not found."})
    return TaskStatusResponse(
        task_id=task.task_id,
        filename=task.filename,
        status=task.status,
        chunks_indexed=task.chunks_indexed,
        error_message=task.error_message,
    )


# --- Q&A ---

@app.post("/ask", response_model=RagAnswer)
@track_latency("ask")
def ask(req: AskRequest):
    """Synchronous Q&A with full response."""
    # Save to session if session_id provided
    if req.session_id:
        store = get_session_store()
        session = store.get_or_create(req.session_id)
        session.add_message("user", req.question)

    result = answer(
        req.question,
        k=req.k,
        filters=filters_to_dict(req.filters),
        session_id=req.session_id,
    )

    # Save assistant response to session
    if req.session_id:
        session.add_message("assistant", result.answer)

    return result


@app.post("/ask/stream")
@track_latency("ask_stream")
def ask_stream(req: AskRequest):
    """
    Streaming Q&A via Server-Sent Events (SSE).
    Yields text chunks as SSE data events.
    """
    from src.stream_batching import get_stream_batcher

    def generate():
        batcher = get_stream_batcher()
        token_stream = answer_stream(
            req.question,
            k=req.k,
            filters=filters_to_dict(req.filters),
        )
        yield from batcher.batch_as_sse(token_stream)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Summarize ---

@app.post("/summarize", response_model=Summary)
@track_latency("summarize")
def summarize_endpoint(req: SummarizeRequest):
    return summarize_learning(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k,
    )


# --- Quiz ---

@app.post("/quiz", response_model=QuizSet)
@track_latency("quiz")
def quiz_endpoint(req: QuizRequest):
    return generate_quiz(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )


# --- Flashcards ---

@app.post("/flashcards", response_model=FlashcardSet)
@track_latency("flashcards")
def flashcards_endpoint(req: FlashcardsRequest):
    return generate_flashcards(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )


# --- Session Management ---

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get conversation history for a session."""
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        return {"session_id": session_id, "messages": []}
    return {"session_id": session_id, "messages": session.get_history()}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clear a session's conversation history."""
    store = get_session_store()
    deleted = store.delete(session_id)
    return {"session_id": session_id, "deleted": deleted}


# --- Feedback ---

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Record thumbs up/down feedback."""
    record_feedback(req.feedback_type)
    return {"status": "recorded", "feedback_type": req.feedback_type}


# --- Observability ---

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)
