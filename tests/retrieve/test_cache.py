from contextguard.retrieve.protocols import MockRetriever
from contextguard.core.specs import SourceType


def test_mock_retriever_cache_returns_same_timestamp():
    # clock that returns deterministic time
    t = "2025-01-01T00:00:00"
    retriever = MockRetriever()
    retriever.enable_cache = True
    retriever._time_fn = lambda: t
    retriever.add_chunk("hello world", source_id="doc1", source_type=SourceType.PRIMARY, entity_ids=["acme"], year=2024)

    first = retriever.search("hello", k=1)
    second = retriever.search("hello", k=1)

    assert len(first) == len(second) == 1
    assert first[0].provenance.retrieved_at == second[0].provenance.retrieved_at == t

