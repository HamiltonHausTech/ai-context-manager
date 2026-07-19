import os

import pytest

from ai_context_manager.embeddings import EmbeddingInfo, EmbeddingProvider
from ai_context_manager.store.postgres_vector_memory import PostgreSQLVectorMemoryStore


pytestmark = pytest.mark.postgres_integration


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


@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 with pgvector available",
)
def test_live_pgvector_search_provenance_and_reembedding():
    provider = IntegrationEmbeddingProvider()
    store = PostgreSQLVectorMemoryStore(
        host="localhost",
        port=5432,
        database="ai_context",
        user="postgres",
        password="demo_password",
        table_name="phase3_vector_test",
        embedding_dimension=3,
        embedding_provider=provider,
    )
    try:
        store.clear_all()
        store.save_component(
            {
                "id": "bank",
                "type": "TaskSummaryComponent",
                "content": "finance bank categorization",
                "tags": ["finance"],
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
                "tags": ["engineering"],
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

        assert results[0]["id"] == "bank"
        assert "old-bank" not in [result["id"] for result in results]
        assert "old-bank" in [
            result["id"]
            for result in store.search_similar(
                "finance bank", n_results=4, include_inactive=True
            )
        ]
        assert results[0]["similarity_score"] == pytest.approx(1.0)
        assert all(0.0 <= result["hybrid_score"] <= 1.0 for result in results)
        assert results[0]["metadata"]["embedding_identifier"] == provider.info.identifier
        assert store.reembed_all(stale_only=True) == 0

        store.embedding_provider = IntegrationEmbeddingProvider("2")
        assert store.reembed_all(stale_only=True) == 4
        assert store.get_embedding_status()["degraded"] is False
    finally:
        store.clear_all()
        store.close()
