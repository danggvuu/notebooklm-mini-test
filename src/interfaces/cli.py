"""
Typer CLI — Command-line interface for Simple NotebookLM v2.0
"""
import json
import typer
from typing import Optional
from src.indexing import ingest as ingest_data_dir
from src.rag import answer, retrieve
from src.learning import summarize as summarize_learning, generate_quiz, generate_flashcards
from src.export import export

app = typer.Typer(help="Simple NotebookLM Command Line Interface v2.0")

def _parse_filters(filters_str: Optional[str]):
    if not filters_str:
        return None
    d = {}
    for part in filters_str.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "page":
                d[k] = int(v)
            else:
                d[k] = v
    return d

def _print_answer(ans: str):
    typer.secho("\nAnswer:", fg=typer.colors.GREEN, bold=True)
    typer.echo(ans)

def _print_sources(chunks):
    typer.secho("\nSources:", fg=typer.colors.BLUE, bold=True)
    for i, c in enumerate(chunks, start=1):
        typer.echo(f"[{i}] {c.metadata.filename} (Page {c.metadata.page}, Score: {c.score:.4f})")
        typer.echo(f"    {c.text[:200]}...")

def _emit(result, output: Optional[str], fmt: str):
    out = export(result, fmt=fmt, output=output)
    if output is None:
        typer.echo(out)
    else:
        typer.secho(f"\nResult exported to {output}", fg=typer.colors.CYAN)

@app.command()
def ingest(recreate: bool = False):
    """Index all documents in the data folder (PDF, DOCX, PPTX, etc.)."""
    typer.echo("Starting document ingestion (MarkItDown + BM25 + Qdrant)...")
    count = ingest_data_dir(recreate=recreate)
    typer.secho(f"Success! Done. {count} chunks indexed.", fg=typer.colors.GREEN)

@app.command()
def ask(
    question: str, 
    k: Optional[int] = None, 
    filters: Optional[str] = None
):
    """Ask a question grounded in the indexed documents (Hybrid Search + Reranker)."""
    parsed = _parse_filters(filters)
    typer.echo(f"Searching index for: '{question}'...")
    result = answer(question, k=k, filters=parsed)
    _print_answer(result.answer)
    _print_sources(result.chunks)

@app.command("debug-retrieval")
def debug_retrieval(
    question: str, 
    k: Optional[int] = None, 
    filters: Optional[str] = None
):
    """Retrieve raw chunks related to a query without calling the LLM."""
    parsed = _parse_filters(filters)
    chunks = retrieve(question, k=k, filters=parsed)
    typer.echo(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))

@app.command("summarize")
def summarize(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    k: Optional[int] = None,
    output: Optional[str] = None,
    fmt: str = "text"
):
    """Generate a multi-point document summary using Map-Reduce."""
    parsed = _parse_filters(filters)
    typer.echo("Generating summary...")
    result = summarize_learning(document=document, query=query, filters=parsed, k=k)
    _emit(result, output, fmt)

@app.command("quiz")
def quiz(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    count: Optional[int] = None,
    k: Optional[int] = None,
    output: Optional[str] = None,
    fmt: str = "text"
):
    """Create a multiple-choice quiz based on document content."""
    parsed = _parse_filters(filters)
    typer.echo("Generating quiz...")
    result = generate_quiz(document=document, query=query, filters=parsed, count=count, k=k)
    _emit(result, output, fmt)

@app.command("flashcards")
def flashcards(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    count: Optional[int] = None,
    k: Optional[int] = None,
    output: Optional[str] = None,
    fmt: str = "text"
):
    """Create interactive flashcards based on document content."""
    parsed = _parse_filters(filters)
    typer.echo("Generating flashcards...")
    result = generate_flashcards(document=document, query=query, filters=parsed, count=count, k=k)
    _emit(result, output, fmt)

@app.command("cache-clear")
def cache_clear():
    """Clear the Redis semantic cache."""
    from src.cache import get_cache
    get_cache().flush_all()
    typer.secho("Cache cleared.", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    app()
