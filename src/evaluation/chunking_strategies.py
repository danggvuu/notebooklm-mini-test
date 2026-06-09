from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
]

_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]

@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: Any
    params: Dict[str, Any]

@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None
    
    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

class SemanticChunkerWrapper:
    """Wrapper for LangChain SemanticChunker."""
    
    def __init__(self, embeddings: Embeddings, breakpoint_type: str = "percentile"):
        self.embeddings = embeddings
        self.breakpoint_type = breakpoint_type
        
    def _splitter(self) -> SemanticChunker:
        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
        )
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)
        
    def split_text(self, text: str) -> List[str]:
        return self._splitter().split_text(text)
