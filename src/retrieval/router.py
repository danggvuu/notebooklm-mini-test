"""
Scope Resolution Router — Tầng Truy xuất
Analyzes query intent to route to the correct retrieval strategy:
- Query-based → HybridSearch (semantic + keyword)
- Full file/card processing → Scroll All Chunks (Qdrant_Scroll)
"""
import logging
from typing import Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ScopeType = Literal["query", "scroll"]


@dataclass
class ScopeDecision:
    """Result of scope resolution."""
    scope_type: ScopeType
    query: Optional[str] = None
    filters: Optional[dict] = None
    reason: str = ""


class ScopeRouter:
    """
    Phân tích Query để quyết định luồng truy xuất.
    - Nếu có query text → HybridSearch (luồng tìm kiếm)
    - Nếu xử lý toàn file/thẻ (summarize, quiz, flashcard toàn bộ) → Scroll All Chunks
    """

    def resolve(
        self,
        query: Optional[str] = None,
        document: Optional[str] = None,
        filters: Optional[dict] = None,
        operation: Optional[str] = None,
    ) -> ScopeDecision:
        """
        Determine the retrieval scope.

        Args:
            query: User's search query (if any)
            document: Specific document filename filter
            filters: Additional metadata filters
            operation: Type of operation ('ask', 'summarize', 'quiz', 'flashcard')
        """
        effective_filters = dict(filters or {})
        if document:
            effective_filters["filename"] = document

        # If there's a query → use hybrid search
        if query and query.strip():
            logger.info("Scope: QUERY → HybridSearch for '%s'", query[:60])
            return ScopeDecision(
                scope_type="query",
                query=query,
                filters=effective_filters or None,
                reason=f"Query detected: '{query[:40]}...'",
            )

        # No query → scroll all chunks (for summarize, quiz, flashcard over entire doc/corpus)
        logger.info("Scope: SCROLL → Full document/corpus retrieval")
        return ScopeDecision(
            scope_type="scroll",
            query=None,
            filters=effective_filters or None,
            reason="No query — scrolling all chunks for full-document operation.",
        )


# Module-level singleton
_router: Optional[ScopeRouter] = None


def get_router() -> ScopeRouter:
    global _router
    if _router is None:
        _router = ScopeRouter()
    return _router
