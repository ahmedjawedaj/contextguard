"""
Qdrant retriever adapter for ContextGuard.

Design (template-method style):
- Wraps a Qdrant client and collection name.
- Uses an embedding function to convert queries to vectors.
- Translates `CanonicalFilters` to Qdrant `Filter` conditions.

Requirements:
- Optional dependency: `qdrant-client`.
- User supplies `embed_fn` (text -> List[float]).

Customization:
- Override `_build_filter` to map more metadata fields.
- Override `_convert_point` to add richer provenance/metadata mapping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import hashlib

from qdrant_client.http import models as qm  # type: ignore  # optional dep

from ..retrieve.protocols import RetrieverBase, CanonicalFilters
from ..core.specs import Chunk, Provenance, SourceType


class QdrantRetrieverAdapter(RetrieverBase):
    """
    Adapter for Qdrant collections.

    Usage:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        adapter = QdrantRetrieverAdapter(
            client=client,
            collection="my_collection",
            embed_fn=my_embed_fn,
        )
        chunks = adapter.search("acme 2024 revenue", filters=CanonicalFilters(...), k=5)
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        embed_fn: Callable[[str], List[float]],
        *,
        source_type: SourceType = SourceType.SECONDARY,
        enable_cache: bool = False,
        time_fn: Optional[Callable[[], str]] = None,
    ):
        super().__init__(name="qdrant", enable_cache=enable_cache, time_fn=time_fn)
        self.client = client
        self.collection = collection
        self.embed_fn = embed_fn
        self.source_type = source_type

    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[CanonicalFilters],
        k: int,
    ) -> List[Chunk]:
        vector = self.embed_fn(query)
        q_filter = self._build_filter(backend_filters)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=k,
            query_filter=q_filter,
        )
        return [self._convert_point(hit) for hit in hits]

    def _build_filter(self, filters: Optional[CanonicalFilters]) -> Optional[qm.Filter]:
        if not filters:
            return None
        must: List[qm.Condition] = []
        if filters.entity_ids:
            must.append(qm.FieldCondition(key="entity_ids", match=qm.MatchAny(any=filters.entity_ids)))
        if filters.year is not None:
            must.append(qm.FieldCondition(key="year", match=qm.MatchValue(value=filters.year)))
        if filters.allowed_source_types:
            must.append(
                qm.FieldCondition(
                    key="source_type",
                    match=qm.MatchAny(any=[st.value for st in filters.allowed_source_types]),
                )
            )
        if filters.doc_types:
            must.append(
                qm.FieldCondition(
                    key="doc_type",
                    match=qm.MatchAny(any=filters.doc_types),
                )
            )
        if not must:
            return None
        return qm.Filter(must=must)

    def _convert_point(self, hit: Any) -> Chunk:
        payload = hit.payload or {}
        text = payload.get("text") or payload.get("content") or ""
        meta: Dict[str, Any] = payload
        source_id = payload.get("source_id") or hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        stype_raw = payload.get("source_type") or self.source_type
        try:
            stype = SourceType(stype_raw)
        except Exception:
            stype = self.source_type

        provenance = Provenance(
            source_id=source_id,
            source_type=stype,
            title=payload.get("title"),
            url=payload.get("url"),
            domain=payload.get("domain"),
            author=payload.get("author"),
            published_at=payload.get("published_at"),
            retrieved_at=payload.get("retrieved_at") or self._time_fn(),
            chunk_id=payload.get("chunk_id") or getattr(hit, "id", None),
        )

        return Chunk(
            text=text,
            score=getattr(hit, "score", None),
            provenance=provenance,
            metadata=meta,
            entity_ids=payload.get("entity_ids", []),
            year=payload.get("year"),
        )

