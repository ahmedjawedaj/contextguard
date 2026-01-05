# Retrievers

Interface
- `Retriever.search(query, filters: CanonicalFilters, k) -> List[Chunk]`
- `Chunk` must include: `text`, `provenance.source_id`, `provenance.source_type`, optional `provenance.url/title/domain`, and metadata: `entity_ids`, `year`, optional `metadata.doc_type`.
- Filters come from `CanonicalFilters.from_state_spec(state)` (entity_ids, year, source types, doc_types, domains).
- Optional async: implement `AsyncRetriever.asearch` to avoid threadpools in async runner.

Provided adapters
- LangChain: `LangChainRetrieverAdapter` (override `doc_to_chunk` or subclass).
- LlamaIndex: `LlamaIndexRetrieverAdapter` (override `node_to_chunk` or subclass).
- Chroma: `ChromaRetrieverAdapter` (embed_fn + metadata filters).
- Qdrant: `QdrantRetrieverAdapter` (embed_fn + filter mapping).
- MockRetriever: for tests/demos.
- Resilience wrappers: `RetryingRetriever` (retry/backoff), `CircuitBreakerRetriever` (trip/half-open/close, optional concurrency guard).
- Dedup/rerank helpers: `dedup_chunks`, `rerank_by_score` (see `retrieve.rerank`).

Implement your own
```python
from contextguard import Retriever
from contextguard.core.specs import Chunk, Provenance, SourceType

class MyRetriever:
    def search(self, query, filters=None, k=10):
        # call your backend, then map results to Chunk
        return [
            Chunk(
                text="...",
                score=0.9,
                provenance=Provenance(source_id="doc1", source_type=SourceType.PRIMARY),
                entity_ids=["acme"],
                year=2024,
                metadata={"doc_type": "annual_report"},
            )
        ]
```

Tips
- Populate `entity_ids` and `year` to let gating work correctly.
- Set `metadata.doc_type` to benefit from diversity and coverage by doc type.
- Use `filters` to push down constraints to your backend when possible; gating will still hard-check.***

