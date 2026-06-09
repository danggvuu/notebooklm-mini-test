"""
Context Builder — Tầng Truy xuất
Đóng gói ngữ cảnh cuối cùng từ reranked chunks hoặc scroll results
cho consumption bởi Jinja2 templates.
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

from src.schemas import RetrievedChunk, Citation

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Packaged context ready for prompt template consumption."""
    chunks: List[RetrievedChunk]
    citations: List[Citation]
    scope: str  # "query", "document", "filter", "corpus"
    target: Optional[str] = None


def format_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """Create citation markers for retrieved chunks."""
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]


class ContextBuilder:
    """
    Đóng gói ngữ cảnh cuối cùng.
    Nhận chunks (từ reranker hoặc scroll) và tạo RetrievalContext
    với citations sẵn sàng cho Jinja2.
    """

    def build(
        self,
        chunks: List[RetrievedChunk],
        scope: str = "query",
        target: Optional[str] = None,
    ) -> RetrievalContext:
        """
        Build retrieval context from chunks.

        Args:
            chunks: Final retrieved/reranked chunks
            scope: "query", "document", "filter", or "corpus"
            target: Description of the target (e.g., query string or filename)
        """
        citations = format_citations(chunks)

        logger.info(
            "Context built: %d chunks, scope=%s, target=%s",
            len(chunks), scope, (target or "")[:40],
        )

        return RetrievalContext(
            chunks=chunks,
            citations=citations,
            scope=scope,
            target=target,
        )


# Module-level singleton
_builder: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    global _builder
    if _builder is None:
        _builder = ContextBuilder()
    return _builder
