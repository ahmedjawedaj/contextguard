import pytest

pytestmark = pytest.mark.optional  # mark as optional; requires chromadb


@pytest.fixture
def chroma_adapter():
    chromadb = pytest.importorskip("chromadb")
    from contextguard import ChromaRetrieverAdapter, SourceType

    client = chromadb.Client()
    collection = client.get_or_create_collection("test_docs")

    # seed a doc
    collection.upsert(
        documents=["ACME 2024 revenue was $200M according to its audited annual report."],
        metadatas=[{"source_id": "d1", "source_type": "PRIMARY", "entity_ids": ["acme"], "year": 2024}],
        ids=["1"],
    )

    def embed(text: str):
        # trivial embedding (not meaningful) for test purposes
        return [float(len(text))]

    adapter = ChromaRetrieverAdapter(collection, embed_fn=embed, source_type=SourceType.PRIMARY)
    return adapter


def test_chroma_search(chroma_adapter):
    from contextguard import CanonicalFilters

    filters = CanonicalFilters(entity_ids=["acme"], year=2024)
    chunks = chroma_adapter.search("ACME revenue 2024", filters=filters, k=3)
    assert len(chunks) >= 1
    assert chunks[0].provenance.source_id == "d1"
    assert "ACME" in chunks[0].text

