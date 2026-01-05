"""
Chroma retriever adapter for ContextGuard.

Design (template-method style):
- Wraps a Chroma collection (client or persistent) and uses metadata filters.
- Converts Chroma results into ContextGuard `Chunk` with full `Provenance`.

Requirements:
- Optional dependency: `chromadb`.
- User must supply an embedding function that maps text -> vector.

Customization:
- Override `_build_query` to change how queries are constructed (e.g., add
  n_results logic).
- Override `_convert_result` to map Chroma documents/metadata to `Chunk`.
- Override `_matches_filters` to add richer filtering beyond Chroma metadata.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import hashlib

from ..retrieve.protocols import RetrieverBase, CanonicalFilters
from ..core.specs import Chunk, Provenance, SourceType


class ChromaRetrieverAdapter(RetrieverBase):
    """
    Adapter for Chroma collections.

    Usage:
        import chromadb
        client = chromadb.Client()
        collection = client.get_collection("my_collection")
        adapter = ChromaRetrieverAdapter(collection, embed_fn=my_embed_fn)
        chunks = adapter.search("acme 2024 revenue", filters=CanonicalFilters(...), k=5)
    """

    def __init__(
        self,
        collection: Any,
        embed_fn: Callable[[str], List[float]],
        *,
        source_type: SourceType = SourceType.SECONDARY,
        enable_cache: bool = False,
        time_fn: Optional[Callable[[], str]] = None,
    ):
        super().__init__(name="chroma", enable_cache=enable_cache, time_fn=time_fn)
        self.collection = collection
        self.embed_fn = embed_fn
        self.source_type = source_type

    def _search_impl(
        self,
        query: str,
        backend_filters: Optional[CanonicalFilters],
        k: int,
    ) -> List[Chunk]:
        query_dict = self._build_query(query, backend_filters, k)
        results = self.collection.query(**query_dict)
        return self._convert_results(results, k)

    def _build_query(self, query: str, filters: Optional[CanonicalFilters], k: int) -> Dict[str, Any]:
        """
        Build Chroma query parameters.
        """
        where: Dict[str, Any] = {}
        if filters:
            if filters.entity_ids:
                where["entity_ids"] = {"$in": filters.entity_ids}
            if filters.year is not None:
                where["year"] = filters.year
            if filters.allowed_source_types:
                where["source_type"] = {"$in": [st.value for st in filters.allowed_source_types]}
            if filters.doc_types:
                where["doc_type"] = {"$in": filters.doc_types}
        return {
            "query_embeddings": [self.embed_fn(query)],
            "where": where or None,
            "n_results": k,
        }

    def _convert_results(self, results: Dict[str, Any], k: int) -> List[Chunk]:
        """
        Convert Chroma query output to Chunks.
        """
        out: List[Chunk] = []
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        scores = results.get("distances", [[]])

        for text, meta, score in zip(docs[0], metas[0], scores[0]):
            meta = meta or {}
            source_id = meta.get("source_id") or meta.get("source") or hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            stype_raw = meta.get("source_type") or self.source_type
            try:
                stype = SourceType(stype_raw)
            except Exception:
                stype = self.source_type

            provenance = Provenance(
                source_id=source_id,
                source_type=stype,
                title=meta.get("title"),
                url=meta.get("url"),
                domain=meta.get("domain"),
                author=meta.get("author"),
                published_at=meta.get("published_at"),
                retrieved_at=meta.get("retrieved_at") or self._time_fn(),
                chunk_id=meta.get("chunk_id"),
            )

            chunk = Chunk(
                text=text,
                score=score,
                provenance=provenance,
                metadata=meta,
                entity_ids=meta.get("entity_ids", []),
                year=meta.get("year"),
            )
            out.append(chunk)
        return out[:k]

