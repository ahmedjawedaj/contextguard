"""
LangChain retriever adapter for ContextGuard.

Design (template-method style):
- Wraps any LangChain retriever that returns `Document` objects.
- Converts each `Document` into a ContextGuard `Chunk` (with full `Provenance`).
- Applies a lightweight post-filter using `CanonicalFilters` when the backend
  cannot apply them natively.

Customization / extension points:
- `doc_to_chunk`: inject your own mapping (e.g., custom provenance fields,
  entity extraction, doc_type normalization).
- Override `_lc_search` to support bespoke retrieval calls.
- Override `_matches_filters` to add richer constraints (e.g., language, tags).
This follows the template-method pattern: `_search_impl` orchestrates; hooks
handle backend-specific behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING
import hashlib

from ..retrieve.protocols import RetrieverBase, CanonicalFilters
from ..core.specs import Chunk, Provenance, SourceType

if TYPE_CHECKING:
    try:
        from langchain_core.documents import Document as LCDocument  # LangChain 0.1+
    except Exception:  # pragma: no cover - optional dependency
        try:
            from langchain.schema import Document as LCDocument  # older LC
        except Exception:
            LCDocument = Any  # type: ignore
else:
    LCDocument = Any  # type: ignore


def _default_doc_to_chunk(
    doc: "LCDocument",
    *,
    source_type: SourceType,
    time_fn: Callable[[], str],
) -> Chunk:
    """
    Default mapping from LangChain `Document` -> `Chunk`.
    Override via `doc_to_chunk` if you need custom provenance/metadata logic.
    """
    meta: Dict[str, Any] = getattr(doc, "metadata", {}) or {}
    text: str = getattr(doc, "page_content", "") or ""

    source_id = (
        meta.get("source_id")
        or meta.get("source")
        or meta.get("document_id")
        or meta.get("id")
        or hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    )
    stype = meta.get("source_type") or source_type
    try:
        stype = SourceType(stype)
    except Exception:
        stype = source_type

    entity_ids = meta.get("entity_ids") or meta.get("entities") or []
    year = meta.get("year")

    provenance = Provenance(
        source_id=source_id,
        source_type=stype,
        title=meta.get("title"),
        url=meta.get("url"),
        domain=meta.get("domain"),
        author=meta.get("author"),
        published_at=meta.get("published_at"),
        retrieved_at=meta.get("retrieved_at") or time_fn(),
        chunk_id=meta.get("chunk_id"),
    )

    return Chunk(
        text=text,
        score=getattr(doc, "score", None),
        provenance=provenance,
        metadata=meta,
        entity_ids=entity_ids if isinstance(entity_ids, list) else [entity_ids],
        year=year,
    )


class LangChainRetrieverAdapter(RetrieverBase):
    """
    Adapter that makes a LangChain retriever conform to ContextGuard's Retriever.

    Typical use:
        from langchain.retrievers import YourRetriever
        lc = YourRetriever(...)
        adapter = LangChainRetrieverAdapter(lc, source_type=SourceType.SECONDARY)
        chunks = adapter.search("acme 2024 revenue", filters=CanonicalFilters(...), k=5)
    """

    def __init__(
        self,
        retriever: Any,
        *,
        source_type: SourceType = SourceType.SECONDARY,
        doc_to_chunk: Optional[
            Callable[["LCDocument", SourceType, Callable[[], str]], Chunk]
        ] = None,
        enable_cache: bool = False,
        time_fn: Optional[Callable[[], str]] = None,
    ):
        super().__init__(name="langchain", enable_cache=enable_cache, time_fn=time_fn)
        self.retriever = retriever
        self.source_type = source_type
        self._doc_to_chunk = doc_to_chunk

    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[CanonicalFilters],
        k: int,
    ) -> List[Chunk]:
        """
        Template method: fetch docs, convert to chunks, apply post-filter.
        Override `_lc_search`, `_convert_doc`, or `_matches_filters` to customize.
        """
        docs = self._lc_search(query, k)
        chunks: List[Chunk] = []
        for doc in docs:
            chunk = self._convert_doc(doc)
            if backend_filters and not self._matches_filters(chunk, backend_filters):
                continue
            chunks.append(chunk)
        return chunks

    def _lc_search(self, query: str, k: int) -> List["LCDocument"]:
        """
        Calls the underlying LangChain retriever. Supports:
        - get_relevant_documents(query)
        - invoke({"query": query}) returning documents
        """
        if hasattr(self.retriever, "get_relevant_documents"):
            return list(self.retriever.get_relevant_documents(query)[:k])
        if hasattr(self.retriever, "invoke"):
            res = self.retriever.invoke({"query": query})
            if isinstance(res, Iterable):
                res_list = list(res)
                return res_list[:k]
        raise TypeError("Retriever must implement get_relevant_documents or invoke")

    def _convert_doc(self, doc: "LCDocument") -> Chunk:
        if self._doc_to_chunk:
            return self._doc_to_chunk(doc, self.source_type, self._time_fn)
        return _default_doc_to_chunk(doc, source_type=self.source_type, time_fn=self._time_fn)

    def _matches_filters(self, chunk: Chunk, filters: CanonicalFilters) -> bool:
        # Entity filter
        if filters.entity_ids:
            if not chunk.entity_ids:
                return False
            if filters.entity_ids_any:
                if not any(eid in filters.entity_ids for eid in chunk.entity_ids):
                    return False
            else:
                if not all(eid in chunk.entity_ids for eid in filters.entity_ids):
                    return False

        # Year filter
        if filters.year is not None and chunk.year is not None and chunk.year != filters.year:
            return False

        # Source type filter
        if filters.allowed_source_types:
            if chunk.provenance.source_type not in filters.allowed_source_types:
                return False

        # Domain filters
        domain = chunk.provenance.domain
        if filters.allowed_domains is not None and domain not in filters.allowed_domains:
            return False
        if filters.blocked_domains is not None and domain in filters.blocked_domains:
            return False

        # Doc type filter (metadata)
        if filters.doc_types:
            doc_type = chunk.metadata.get("doc_type")
            if doc_type is None or doc_type not in filters.doc_types:
                return False

        return True
