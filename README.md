# Simple NotebookLM 📓 v2.0

Simple NotebookLM is a grounded learning assistant based on a **production-grade Retrieval-Augmented Generation (RAG)** architecture. It allows you to upload personal textbooks, articles, or documents in multiple formats (PDF, DOCX, PPTX, XLSX, HTML, etc.), automatically index them using a **hybrid search pipeline** (Vector + Keyword), and then interact with the content through smart Q&A with exact citations, Map-Reduce summaries, and automatically generated interactive study materials (Quizzes and Flashcards).

This project is fully implemented based on the **AI VIET NAM (AIO2025)** curriculum specification.

---

## 🏗️ System Architecture (Flowchart TB)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tầng Giao diện & Định tuyến API                                   │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │  Streamlit Web UI    │◄─►│  FastAPI Backend (SSE Support)  │   │
│  │  Workspace-based     │  │  REST API + SSE Stream           │   │
│  └──────────────────────┘  └──────────┬───────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────┐
│  Tầng Vận hành & Giám sát (MLOps)                                  │
│  ┌─────────────┐  ┌────────────────┐  ┌─────────────────────────┐ │
│  │ Redis Cache  │  │ Session Memory │  │ Prometheus + LangSmith  │ │
│  │ ~0.1s reply  │  │ Chat History   │  │ Tracing & Feedback      │ │
│  └─────────────┘  └────────────────┘  └─────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────┐
│  Tầng Xử lý Dữ liệu Bất đồng bộ                                   │
│  ┌────────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Background     │─►│ MarkItDown    │─►│ Recursive Chunker    │  │
│  │ Worker (FastAPI│  │ Parser (OCR)  │  │ 1000 size / 150 olap │  │
│  │ BackgroundTask)│  └───────────────┘  └──────────┬───────────┘  │
│  └────────────────┘                                │              │
│                                    ┌───────────────┼──────────┐   │
│                                    ▼               ▼          │   │
│                            ┌──────────────┐ ┌────────────┐    │   │
│                            │ GreenNode    │ │ RankBM25   │    │   │
│                            │ Embedding    │ │ Inv. Index │    │   │
│                            └──────┬───────┘ └─────┬──────┘    │   │
└───────────────────────────────────┼───────────────┼───────────────┘
                                    │               │
┌───────────────────────────────────▼───────────────▼───────────────┐
│  Kho Tri thức (Data Isolation)                                    │
│  ┌────────────────────────┐  ┌────────────────────────────────┐  │
│  │ Qdrant Vector DB       │  │ RankBM25 (RAM Inverted Index)  │  │
│  │ Payload Index          │  │ Persisted to disk              │  │
│  └────────────────────────┘  └────────────────────────────────┘  │
└──────────────────────────────────┬────────────────────────────────┘
                                   │
┌──────────────────────────────────▼────────────────────────────────┐
│  Tầng Truy xuất Lai & Lọc nhiễu                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Scope Router │─►│ Hybrid Search│─►│ Cross-Encoder        │   │
│  │ Query Analy. │  │ (Parallel)   │  │ Reranker (BGE-v2-m3) │   │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘   │
│                                                  ▼               │
│                                       ┌──────────────────────┐   │
│                                       │ Context Builder      │   │
│                                       └──────────┬───────────┘   │
└──────────────────────────────────────────────────┼───────────────┘
                                                   │
┌──────────────────────────────────────────────────▼───────────────┐
│  Tầng Tạo sinh & Kiểm duyệt                                     │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────┐ │
│  │ Jinja2      │─►│ LLM Factory   │─►│ Stream Batching (50ms) │ │
│  │ Templates   │  │ vLLM/HF/Gemini│  │ → SSE to Frontend      │ │
│  └─────────────┘  └───────────────┘  └────────────────────────┘ │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │ Map-Reduce Batching  │  │ Pydantic Validation              │ │
│  │ Long-doc summarize   │  │ JSON type enforcement            │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features

1. **Grounded Q&A (Hỏi đáp có trích dẫn)**: Answers user questions using **Hybrid Search** (Semantic + Keyword) with **Cross-Encoder Reranking**. Traces all sentences back to sources using precise `[S1]`, `[S2]` markers.
2. **Streaming Responses (SSE)**: Real-time token streaming via Server-Sent Events with configurable token batching.
3. **Redis Semantic Cache**: Sub-100ms responses for repeated/similar queries.
4. **Map-Reduce Summarization**: Dynamically summarizes extremely long texts by chunking, summarizing individually, and aggregating results.
5. **Interactive Quizzes & Flashcards**: Generates study materials with Pydantic-validated JSON output.
6. **Multimodal Document Support**: MarkItDown parser handles PDF, DOCX, PPTX, XLSX, HTML, Markdown, and images with OCR.
7. **Async Ingestion**: Background file processing with status tracking.
8. **Observability**: Prometheus metrics + optional LangSmith tracing + user feedback (thumbs up/down).
9. **Multi-interface Support**:
   - **Streamlit Web UI**: Premium glassmorphism dark-theme dashboard with workspace isolation.
   - **FastAPI REST API**: Decoupled core endpoints with SSE streaming.
   - **Typer CLI**: Direct command-line automation.
10. **RAG Evaluation Platform**: Built-in benchmark harness with `ragas` metrics.

---

## 📂 Project Structure

```
./
├── data/                          # Document Input folder (PDF, DOCX, etc.)
├── storage/
│   ├── qdrant/                    # Qdrant vector DB storage
│   └── bm25/                     # BM25 index persistence
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic Settings (extended)
│   ├── schemas.py                 # Pydantic data schemas
│   ├── cache.py                   # Redis Semantic Cache
│   ├── session.py                 # Short-term Conversation Memory
│   ├── observability.py           # Prometheus + LangSmith
│   ├── indexing.py                # MarkItDown parser + chunking
│   ├── worker.py                  # Background Worker (FastAPI BackgroundTasks)
│   ├── bm25_index.py              # RankBM25 inverted index
│   ├── store.py                   # Qdrant client
│   ├── retrieval/                 # Retrieval pipeline
│   │   ├── __init__.py
│   │   ├── router.py             # Scope Resolution
│   │   ├── hybrid_search.py      # Hybrid Search (Qdrant + BM25)
│   │   ├── reranker.py           # Cross-Encoder Reranker
│   │   └── context_builder.py    # Context packaging
│   ├── rag.py                     # Main RAG orchestrator
│   ├── llm.py                     # LLM Factory Pattern (vLLM/HF/Gemini)
│   ├── stream_batching.py         # Token Buffer for SSE
│   ├── learning.py                # Summarize/Quiz/Flashcards
│   ├── filters.py                 # Metadata filters
│   ├── export.py                  # Export formats
│   ├── prompts/                   # Jinja2 templates
│   │   ├── answer.jinja2
│   │   ├── summary_*.jinja2
│   │   ├── quiz.jinja2
│   │   └── flashcards.jinja2
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── api.py                # FastAPI + SSE
│   │   ├── cli.py                # Typer CLI
│   │   ├── styles.py             # CSS theme
│   │   └── ui.py                 # Streamlit UI
│   └── evaluation/               # Ragas benchmarks
│       ├── __init__.py
│       ├── benchmark_rag.csv
│       ├── ragas_evaluator.py
│       ├── chunking_strategies.py
│       ├── run_chunking.py
│       └── run_reranking.py
├── docker-compose.yml             # Redis server
├── .env
├── .env.example
├── requirements.txt
├── flowchart TB.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

### 2. Start Redis (Docker)
```bash
docker compose up -d
```
This starts a Redis server on port 6379 with persistence. If Redis is unavailable, the system will run without caching.

### 3. Configure Environment Parameters
Create a copy of `.env.example` as `.env` and configure your API keys:
```bash
cp .env.example .env
```
Open `.env` and fill in:
- `GOOGLE_API_KEY`: Your Google Gemini API Key (recommended backend).
- `RAG_LLM_PROVIDER`: Set to `gemini` (default) or `hf_local`/`vllm`.
- `RAG_EMBEDDING_MODEL`: Defaults to `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`.

---

## 🚀 Running the Application

### 1. Start the FastAPI Backend
```bash
uvicorn src.interfaces.api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Streamlit Web UI
In a separate terminal tab:
```bash
streamlit run src/interfaces/ui.py
```
Open your browser at `http://localhost:8501` to view the dashboard!

### 3. Monitor Metrics
- **Prometheus**: `http://localhost:8000/metrics`
- **Health**: `http://localhost:8000/health`

---

## 💻 Typer CLI Automation

### 1. Index All Documents
Place your documents in the `data/` folder and run:
```bash
python -m src.interfaces.cli ingest
```

### 2. Ask Grounded Questions (Hybrid Search + Reranker)
```bash
python -m src.interfaces.cli ask "LoRA là gì?"
```

### 3. Generate Summaries, Quizzes, and Flashcards
```bash
python -m src.interfaces.cli summarize --fmt md --output storage/summary.md
python -m src.interfaces.cli quiz --count 5
python -m src.interfaces.cli flashcards --count 8
```

### 4. Clear Cache
```bash
python -m src.interfaces.cli cache-clear
```

---

## 📊 Running Evaluations (Ragas)

### 1. Run Chunking Evaluation
```bash
python -m src.evaluation.run_chunking
```

### 2. Run Reranking Evaluation
```bash
python -m src.evaluation.run_reranking
```
All outputs are saved as structured JSON matrices in `storage/evaluation/` for analysis.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/documents` | GET | List indexed documents |
| `/upload` | POST | Upload file (async, returns task_id) |
| `/upload/status/{task_id}` | GET | Check upload task status |
| `/ask` | POST | Q&A (synchronous, full response) |
| `/ask/stream` | POST | Q&A (streaming via SSE) |
| `/summarize` | POST | Generate summary (Map-Reduce) |
| `/quiz` | POST | Generate quiz |
| `/flashcards` | POST | Generate flashcards |
| `/session/{session_id}` | GET | Get conversation history |
| `/session/{session_id}` | DELETE | Clear conversation history |
| `/feedback` | POST | Record thumbs up/down |
| `/metrics` | GET | Prometheus metrics |
