"""
LlamaIndex retriever adapter for ContextGuard.

Design (template-method style):
- Wraps any LlamaIndex retriever/query engine exposing `.retrieve(query)`.
- Converts `NodeWithScore` results into ContextGuard `Chunk` with `Provenance`.
- Applies lightweight post-filtering via `CanonicalFilters` when the backend
  cannot apply them natively.

Customization / extension points:
- `node_to_chunk`: inject custom mapping (provenance, metadata normalization).
- Override `_li_search` to support custom retrieval calls.
- Override `_matches_filters` to enforce richer constraints.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import hashlib

from ..retrieve.protocols import RetrieverBase, CanonicalFilters
from ..core.specs import Chunk, Provenance, SourceType

if TYPE_CHECKING:
    try:
        from llama_index.core.schema import NodeWithScore  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        NodeWithScore = Any  # type: ignore
else:
    NodeWithScore = Any  # type: ignore


def _default_node_to_chunk(
    node_with_score: "NodeWithScore",
    *,
    source_type: SourceType,
) -> Chunk:
    """
    Default mapping from LlamaIndex `NodeWithScore` -> `Chunk`.
    Override via `node_to_chunk` for custom metadata/provenance handling.
    """
    node = getattr(node_with_score, "node", node_with_score)
    text = ""
    if hasattr(node, "get_content"):
        try:
            text = node.get_content(metadata_mode="all")
        except Exception:
            text = node.get_content()
    text = text or getattr(node, "text", "") or ""

    meta: Dict[str, Any] = getattr(node, "metadata", {}) or {}
    score = getattr(node_with_score, "score", None)

    source_id = (
        meta.get("source_id")
        or meta.get("source")
        or meta.get("document_id")
        or meta.get("id")
        or getattr(node, "node_id", None)
        or hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    )
    stype = meta.get("source_type") or source_type
    try:
        stype = SourceType(stype)
    except Exception:
        stype = source_type

    entity_ids = meta.get("entity_ids") or []
    year = meta.get("year")

    provenance = Provenance(
        source_id=source_id,
        source_type=stype,
        title=meta.get("title"),
        url=meta.get("url"),
        domain=meta.get("domain"),
        author=meta.get("author"),
        published_at=meta.get("published_at"),
        retrieved_at=meta.get("retrieved_at"),
        chunk_id=meta.get("chunk_id") or getattr(node, "node_id", None),
    )

    return Chunk(
        text=text,
        score=score,
        provenance=provenance,
        metadata=meta,
        entity_ids=entity_ids if isinstance(entity_ids, list) else [entity_ids],
        year=year,
    )


class LlamaIndexRetrieverAdapter(RetrieverBase):
    """
    Adapter that makes a LlamaIndex retriever conform to ContextGuard's Retriever.

    Typical use:
        li = index.as_retriever()
        adapter = LlamaIndexRetrieverAdapter(li, source_type=SourceType.PRIMARY)
        chunks = adapter.search("acme 2024 revenue", filters=CanonicalFilters(...), k=5)
    """

    def __init__(
        self,
        retriever: Any,
        *,
        source_type: SourceType = SourceType.SECONDARY,
        node_to_chunk: Optional[Callable[[Any], Chunk]] = None,
    ):
        super().__init__(name="llamaindex")
        self.retriever = retriever
        self.source_type = source_type
        self._node_to_chunk = node_to_chunk

    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[CanonicalFilters],
        k: int,
    ) -> List[Chunk]:
        """
        Template method: fetch nodes, convert to chunks, apply post-filter.
        Override `_li_search`, `_convert_node`, or `_matches_filters` to customize.
        """
        nodes = self._li_search(query, k)
        chunks: List[Chunk] = []
        for node in nodes:
            chunk = self._convert_node(node)
            if backend_filters and not self._matches_filters(chunk, backend_filters):
                continue
            chunks.append(chunk)
        return chunks

    def _li_search(self, query: str, k: int) -> List[Any]:
        if hasattr(self.retriever, "retrieve"):
            results = self.retriever.retrieve(query)
            if isinstance(results, list):
                return results[:k]
            if hasattr(results, "__iter__"):
                return list(results)[:k]
        raise TypeError("Retriever must implement retrieve(query)")

    def _convert_node(self, node_with_score: Any) -> Chunk:
        if self._node_to_chunk:
            return self._node_to_chunk(node_with_score)
        return _default_node_to_chunk(node_with_score, source_type=self.source_type)

    def _matches_filters(self, chunk: Chunk, filters: CanonicalFilters) -> bool:
        if filters.entity_ids:
            if not chunk.entity_ids:
                return False
            if filters.entity_ids_any:
                if not any(eid in filters.entity_ids for eid in chunk.entity_ids):
                    return False
            else:
                if not all(eid in chunk.entity_ids for eid in filters.entity_ids):
                    return False

        if filters.year is not None and chunk.year is not None and chunk.year != filters.year:
            return False

        if filters.allowed_source_types:
            if chunk.provenance.source_type not in filters.allowed_source_types:
                return False

        if filters.doc_types:
            doc_type = chunk.metadata.get("doc_type")
            if doc_type is None or doc_type not in filters.doc_types:
                return False

        return True
