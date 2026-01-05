import pytest

pytestmark = pytest.mark.optional  # mark as optional; requires qdrant-client


@pytest.fixture
def qdrant_adapter(tmp_path):
    qdrant_client = pytest.importorskip("qdrant_client")
    from contextguard import QdrantRetrieverAdapter, SourceType

    client = qdrant_client.QdrantClient(":memory:")
    collection = "test_docs"
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qdrant_client.http.models.VectorParams(size=1, distance="Cosine"),
    )

    # upsert a point
    client.upsert(
        collection_name=collection,
        points=[
            qdrant_client.http.models.PointStruct(
                id=1,
                vector=[1.0],
                payload={
                    "text": "ACME 2024 revenue was $200M according to its audited annual report.",
                    "source_id": "d1",
                    "source_type": "PRIMARY",
                    "entity_ids": ["acme"],
                    "year": 2024,
                },
            )
        ],
    )

    def embed(text: str):
        return [1.0]  # trivial embedding

    adapter = QdrantRetrieverAdapter(
        client=client,
        collection=collection,
        embed_fn=embed,
        source_type=SourceType.PRIMARY,
    )
    return adapter


def test_qdrant_search(qdrant_adapter):
    from contextguard import CanonicalFilters

    filters = CanonicalFilters(entity_ids=["acme"], year=2024)
    chunks = qdrant_adapter.search("ACME 2024 revenue", filters=filters, k=3)
    assert len(chunks) >= 1
    assert chunks[0].provenance.source_id == "d1"
    assert "ACME" in chunks[0].text

