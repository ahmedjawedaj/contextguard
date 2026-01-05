"""
ContextGuard Retrieval Protocols

This module defines the universal interfaces for retrieval that work across
any vector database, search backend, or hybrid system.

Key abstraction: ContextGuard doesn't care HOW you retrieve - it cares WHAT
you retrieve and whether it satisfies the current StateSpec.

Protocols defined here:
- Retriever: The universal retrieval interface
- CanonicalFilters: Backend-agnostic filter specification
- Chunk: Universal chunk representation (see core/specs.py)

Adapters in contextguard/adapters/ translate these to specific backends:
- LangChain retrievers
- LlamaIndex retrievers
- Direct pgvector/Qdrant/Chroma/Weaviate calls
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Callable, Tuple
import json
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone

from ..core.specs import Chunk, SourceType, Provenance


# =============================================================================
# CANONICAL FILTERS: Backend-agnostic filter specification
# =============================================================================


class CanonicalFilters(BaseModel):
    """
    Universal filter specification for retrieval.
    
    This is translated by adapters into backend-specific filter syntax
    (Qdrant filter dict, pgvector WHERE clause, Chroma where, etc.).
    
    Design principle: express filters in domain terms (entity, time, source),
    not in vector DB terms (metadata.field == value).
    """
    model_config = ConfigDict(extra="forbid")
    
    # Entity constraints
    entity_ids: List[str] = Field(default_factory=list)
    entity_ids_any: bool = True  # True = OR, False = AND
    
    # Time constraints
    year: Optional[int] = None
    quarter: Optional[int] = None
    start_date: Optional[str] = None  # ISO date: "YYYY-MM-DD"
    end_date: Optional[str] = None
    fiscal: Optional[bool] = None
    
    # Source constraints
    allowed_source_types: List[SourceType] = Field(default_factory=list)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    max_age_days: Optional[int] = None
    
    # Document type constraints (10-K, earnings_call, etc.)
    doc_types: Optional[List[str]] = None
    
    # Language constraint
    language: Optional[str] = None
    
    # Arbitrary metadata constraints (adapter decides support)
    # Use for backend-specific filters
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            not self.entity_ids
            and self.year is None
            and self.quarter is None
            and self.start_date is None
            and self.end_date is None
            and not self.allowed_source_types
            and self.allowed_domains is None
            and self.blocked_domains is None
            and self.max_age_days is None
            and self.doc_types is None
            and self.language is None
            and not self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True, exclude_defaults=True)
    
    @classmethod
    def from_state_spec(cls, state) -> "CanonicalFilters":
        """
        Create filters from a StateSpec.
        
        This is the main translation from state contract to retrieval filters.
        """
        from ..core.specs import StateSpec
        
        if not isinstance(state, StateSpec):
            raise TypeError(f"Expected StateSpec, got {type(state)}")
        
        filters = cls()
        
        # Entity filters
        if state.entities:
            filters.entity_ids = [e.entity_id for e in state.entities]
        
        # Time filters
        if state.time.year is not None:
            filters.year = state.time.year
        if state.time.quarter is not None:
            filters.quarter = state.time.quarter
        if state.time.start_date is not None:
            filters.start_date = state.time.start_date
        if state.time.end_date is not None:
            filters.end_date = state.time.end_date
        if state.time.fiscal:
            filters.fiscal = state.time.fiscal
        
        # Source policy filters
        if state.source_policy.allowed_source_types:
            filters.allowed_source_types = state.source_policy.allowed_source_types
        if state.source_policy.allowed_domains is not None:
            filters.allowed_domains = state.source_policy.allowed_domains
        if state.source_policy.blocked_domains is not None:
            filters.blocked_domains = state.source_policy.blocked_domains
        if state.source_policy.max_age_days is not None:
            filters.max_age_days = state.source_policy.max_age_days
        
        # Language
        if state.language:
            filters.language = state.language
        
        return filters


# =============================================================================
# RETRIEVER PROTOCOL: The universal retrieval interface
# =============================================================================


@runtime_checkable
class Retriever(Protocol):
    """
    Protocol for any retrieval backend.
    
    Implementations:
    - contextguard.adapters.langchain.LangChainRetriever
    - contextguard.adapters.llamaindex.LlamaIndexRetriever
    - Direct implementations for pgvector, Qdrant, Chroma, etc.
    
    The only method required is search(). Everything else is optional.
    """
    
    def search(
        self,
        query: str,
        *,
        filters: Optional[CanonicalFilters] = None,
        k: int = 10,
    ) -> List[Chunk]:
        """
        Search for chunks matching the query.
        
        Args:
            query: The search query (natural language)
            filters: Optional filters to apply
            k: Maximum number of results to return
            
        Returns:
            List of Chunk objects with provenance
        """
        ...


@runtime_checkable
class AsyncRetriever(Protocol):
    """
    Optional async retriever interface.
    
    If implemented, async runners can call `asearch` directly instead of using thread pools.
    """
    
    async def asearch(
        self,
        query: str,
        *,
        filters: Optional[CanonicalFilters] = None,
        k: int = 10,
    ) -> List[Chunk]:
        ...


class RetrieverBase(ABC):
    """
    Base class for retriever implementations.
    
    Provides common functionality like filter translation and logging.
    Subclasses must implement _search_impl().
    """
    
    def __init__(
        self,
        name: str = "base",
        default_k: int = 10,
        enable_cache: bool = False,
        time_fn: Optional[Callable[[], str]] = None,
    ):
        self.name = name
        self.default_k = default_k
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple[str, str, str, int], List[Chunk]] = {}
        # time_fn should return ISO string if provided
        self._time_fn = time_fn
    
    def search(
        self,
        query: str,
        *,
        filters: Optional[CanonicalFilters] = None,
        k: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Public search method with common pre/post processing.
        """
        k = k or self.default_k
        
        # Pre-process filters
        backend_filters = self._translate_filters(filters) if filters else None
        cache_key = None
        if self.enable_cache:
            cache_key = (
                self.name,
                query,
                json.dumps(filters.to_dict() if filters else {}, sort_keys=True),
                k,
            )
            if cache_key in self._cache:
                return [Chunk.model_validate(cdict) for cdict in self._cache[cache_key]]
        
        # Perform search
        chunks = self._search_impl(query, backend_filters, k)
        
        # Ensure all chunks have provenance
        for chunk in chunks:
            if chunk.provenance.retrieved_at is None:
                if self._time_fn:
                    chunk.provenance.retrieved_at = self._time_fn()
                else:
                    chunk.provenance.retrieved_at = datetime.now(timezone.utc).isoformat()
            if chunk.provenance.retrieval_query is None:
                chunk.provenance.retrieval_query = query

        if self.enable_cache and cache_key is not None:
            # store deep copies via model_dump
            self._cache[cache_key] = [json.loads(c.model_dump_json()) for c in chunks]
        
        return chunks
    
    @abstractmethod
    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[Any],
        k: int,
    ) -> List[Chunk]:
        """
        Subclasses implement the actual search logic here.
        """
        ...
    
    def _translate_filters(
        self,
        filters: CanonicalFilters,
    ) -> Any:
        """
        Translate canonical filters to backend-specific format.
        
        Override in subclasses for backend-specific translation.
        Default: return the canonical filters as-is.
        """
        return filters


# =============================================================================
# MOCK RETRIEVER: For testing and demos
# =============================================================================


class MockRetriever(RetrieverBase):
    """
    Mock retriever for testing.
    
    Pre-loaded with chunks that can be searched.
    Useful for unit tests and demos without a real vector DB.
    """
    
    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        name: str = "mock",
    ):
        super().__init__(name=name)
        self.chunks: List[Chunk] = chunks or []
    
    def add_chunk(
        self,
        text: str,
        source_id: str = "mock_doc",
        source_type: SourceType = SourceType.SECONDARY,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a chunk to the mock store."""
        chunk = Chunk(
            text=text,
            score=1.0,
            provenance=Provenance(
                source_id=source_id,
                source_type=source_type,
            ),
            entity_ids=entity_ids or [],
            year=year,
            metadata=metadata or {},
        )
        self.chunks.append(chunk)
    
    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[CanonicalFilters],
        k: int,
    ) -> List[Chunk]:
        """
        Simple keyword matching + filter application.
        """
        results = []
        query_lower = query.lower()
        
        for chunk in self.chunks:
            # Simple relevance: keyword overlap
            text_lower = chunk.text.lower()
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)
            
            if overlap == 0:
                continue
            
            # Apply filters
            if backend_filters:
                if not self._matches_filters(chunk, backend_filters):
                    continue
            
            # Score by overlap
            score = overlap / len(query_words) if query_words else 0.0
            
            # Create result chunk with score
            result = Chunk(
                text=chunk.text,
                score=score,
                provenance=chunk.provenance,
                entity_ids=chunk.entity_ids,
                year=chunk.year,
                metadata=chunk.metadata,
            )
            results.append((score, result))
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in results[:k]]
    
    def _matches_filters(
        self,
        chunk: Chunk,
        filters: CanonicalFilters,
    ) -> bool:
        """Check if chunk matches all filters."""
        
        # Entity filter
        if filters.entity_ids:
            if not chunk.entity_ids:
                return False
            if filters.entity_ids_any:
                # OR: any match is fine
                if not any(eid in filters.entity_ids for eid in chunk.entity_ids):
                    return False
            else:
                # AND: all must match
                if not all(eid in chunk.entity_ids for eid in filters.entity_ids):
                    return False
        
        # Year filter
        if filters.year is not None:
            if chunk.year is None or chunk.year != filters.year:
                return False
        
        # Source type filter
        if filters.allowed_source_types:
            if chunk.provenance.source_type not in filters.allowed_source_types:
                return False
        
        # Domain filters
        if filters.blocked_domains and chunk.provenance.domain:
            if chunk.provenance.domain in filters.blocked_domains:
                return False
        
        if filters.allowed_domains is not None and chunk.provenance.domain:
            if chunk.provenance.domain not in filters.allowed_domains:
                return False
        
        return True


# =============================================================================
# FEDERATED RETRIEVER: Combine multiple sources
# =============================================================================


class FederatedRetriever(RetrieverBase):
    """
    Retriever that combines results from multiple backends.
    
    Useful for:
    - Corpus + web retrieval
    - Multiple corpora (internal + external)
    - Hybrid search (vector + BM25)
    """
    
    def __init__(
        self,
        retrievers: List[Retriever],
        name: str = "federated",
        merge_strategy: str = "interleave",  # "interleave", "concat", "score_sort"
    ):
        super().__init__(name=name)
        self.retrievers = retrievers
        self.merge_strategy = merge_strategy
    
    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[Any],
        k: int,
    ) -> List[Chunk]:
        """Search all retrievers and merge results."""
        
        all_results: List[Chunk] = []
        
        # Search each retriever
        per_retriever_k = max(k // len(self.retrievers), 5)
        
        for retriever in self.retrievers:
            try:
                chunks = retriever.search(
                    query,
                    filters=backend_filters,
                    k=per_retriever_k,
                )
                all_results.extend(chunks)
            except Exception:
                # Log but continue with other retrievers
                # In production, you'd want proper logging here
                pass
        
        # Merge based on strategy
        if self.merge_strategy == "score_sort":
            # Sort all by score
            all_results.sort(key=lambda c: c.score or 0.0, reverse=True)
        elif self.merge_strategy == "interleave":
            # Round-robin interleaving (maintains source diversity)
            all_results = self._interleave(all_results, len(self.retrievers))
        # "concat" just keeps them in retriever order
        
        return all_results[:k]
    
    def _interleave(self, chunks: List[Chunk], num_sources: int) -> List[Chunk]:
        """Interleave chunks from different sources."""
        # Group by source
        by_source: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            source = chunk.provenance.source_id
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        # Round-robin
        result = []
        sources = list(by_source.keys())
        idx = 0
        
        while len(result) < len(chunks):
            source = sources[idx % len(sources)]
            if by_source[source]:
                result.append(by_source[source].pop(0))
            idx += 1
            
            # Safety: break if all sources exhausted
            if all(len(v) == 0 for v in by_source.values()):
                break
        
        return result
