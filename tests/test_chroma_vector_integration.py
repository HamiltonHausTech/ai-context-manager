import pytest

from ai_context_manager.embeddings import EmbeddingInfo, EmbeddingProvider
from ai_context_manager.store.vector_memory import CHROMADB_AVAILABLE, VectorMemoryStore


pytestmark = pytest.mark.chroma_integration


class IntegrationEmbeddingProvider(EmbeddingProvider):
    def __init__(self, version="1"):
        self.version = version

    @property
    def info(self):
        return EmbeddingInfo("test", "integration-keywords", self.version, 3)

    def embed(self, text):
        text = text.lower()
        return [
            float("finance" in text or "bank" in text),
            float("weather" in text or "rain" in text),
            float("python" in text or "code" in text),
        ]


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB is not installed")
def test_live_chroma_search_provenance_filters_and_reembedding(tmp_path):
    provider = IntegrationEmbeddingProvider()
    store = VectorMemoryStore(
        collection_name="phase3_vector_test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_provider=provider,
    )
    store.save_component(
        {
            "id": "bank",
            "type": "TaskSummaryComponent",
            "content": "finance bank categorization",
            "tags": ["finance", "work"],
            "score": 1.0,
        }
    )
    store.save_component(
        {
            "id": "rain",
            "type": "TaskSummaryComponent",
            "content": "weather and rain forecast",
            "tags": ["weather"],
            "score": 1.0,
        }
    )
    store.save_component(
        {
            "id": "code",
            "type": "TaskSummaryComponent",
            "content": "python code review",
            "tags": ["engineering", "work"],
            "score": 1.0,
        }
    )
    store.save_component(
        {
            "id": "old-bank",
            "type": "TaskSummaryComponent",
            "content": "finance bank obsolete rules",
            "tags": ["finance"],
            "score": 1.0,
            "memory": {"kind": "episode", "status": "superseded"},
        }
    )

    results = store.search_similar("finance bank", n_results=4)
    work_results = store.search_similar(
        "finance bank", n_results=3, include_tags=["work"]
    )

    assert results[0]["id"] == "bank"
    assert "old-bank" not in [result["id"] for result in results]
    assert "old-bank" in [
        result["id"]
        for result in store.search_similar(
            "finance bank", n_results=4, include_inactive=True
        )
    ]
    assert results[0]["similarity_score"] == pytest.approx(1.0)
    assert [result["id"] for result in work_results] == ["bank", "code"]
    assert store.collection.get(ids=["bank"])["metadatas"][0][
        "embedding_identifier"
    ] == provider.info.identifier
    assert store.reembed_all(stale_only=True) == 0

    store.embedding_provider = IntegrationEmbeddingProvider("2")
    assert store.reembed_all(stale_only=True) == 4
    assert store.get_embedding_status()["degraded"] is False
