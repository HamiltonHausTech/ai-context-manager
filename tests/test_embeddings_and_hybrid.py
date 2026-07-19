from datetime import datetime, timezone

import pytest

from ai_context_manager.embeddings import (
    EmbeddingDimensionError,
    EmbeddingInfo,
    EmbeddingProvider,
    create_embedding_text,
    embedding_metadata,
    needs_reembedding,
    stable_content_hash,
)
from ai_context_manager.hybrid import HybridWeights, hybrid_score, rank_hybrid
from ai_context_manager.semantic_context_manager import SemanticContextManager
from ai_context_manager.store.base import MemoryStore
from ai_context_manager.store.postgres_vector_memory import PostgreSQLVectorMemoryStore
from ai_context_manager.store.vector_memory import VectorMemoryStore


class KeywordEmbeddingProvider(EmbeddingProvider):
    def __init__(self, version="1"):
        self.version = version

    @property
    def info(self):
        return EmbeddingInfo("test", "keywords", self.version, 3)

    def embed(self, text):
        lowered = text.lower()
        return [
            float("finance" in lowered or "bank" in lowered),
            float("weather" in lowered or "rain" in lowered),
            float("python" in lowered or "code" in lowered),
        ]


def test_content_hash_is_stable_and_sensitive_to_content():
    assert stable_content_hash("same") == stable_content_hash("same")
    assert stable_content_hash("same") != stable_content_hash("different")
    assert len(stable_content_hash("same")) == 64


def test_provider_dimension_is_enforced():
    class BrokenProvider(KeywordEmbeddingProvider):
        def embed(self, _text):
            return [1.0]

    with pytest.raises(EmbeddingDimensionError, match="expected 3"):
        BrokenProvider().embed_checked("finance")


def test_embedding_metadata_detects_content_and_provider_changes():
    provider = KeywordEmbeddingProvider("1")
    metadata = embedding_metadata(provider, "finance")

    assert not needs_reembedding(metadata, provider, "finance")
    assert needs_reembedding(metadata, provider, "weather")
    assert needs_reembedding(metadata, KeywordEmbeddingProvider("2"), "finance")


def test_vector_backends_share_the_same_provider_contract():
    provider = KeywordEmbeddingProvider()
    chroma_store = VectorMemoryStore.__new__(VectorMemoryStore)
    chroma_store.embedding_provider = provider
    chroma_store.last_embedding_error = None
    postgres_store = PostgreSQLVectorMemoryStore.__new__(PostgreSQLVectorMemoryStore)
    postgres_store.embedding_provider = provider
    postgres_store.last_embedding_error = None

    assert chroma_store._generate_embedding("python code") == [0.0, 0.0, 1.0]
    assert postgres_store._generate_embedding("python code") == [0.0, 0.0, 1.0]


def test_hybrid_ranking_uses_semantics_importance_recency_and_feedback():
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    records = [
        {
            "id": "semantic",
            "similarity_score": 0.95,
            "score": 0.2,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "feedback_score": 0.0,
        },
        {
            "id": "important",
            "similarity_score": 0.70,
            "score": 5.0,
            "timestamp": "2026-07-18T00:00:00+00:00",
            "feedback_score": 1.0,
        },
    ]
    weights = HybridWeights(semantic=0.4, importance=0.3, recency=0.2, feedback=0.1)

    ranked = rank_hybrid(records, weights, now)

    assert [record["id"] for record in ranked] == ["important", "semantic"]
    assert all(0.0 <= record["hybrid_score"] <= 1.0 for record in ranked)


def test_backend_independent_relevance_fixture_has_same_ordering():
    # Chroma cosine distance and pgvector cosine similarity are normalized by
    # their adapters before reaching this shared ranker.
    fixture = [
        {"id": "bank", "similarity_score": 0.9, "score": 1.0},
        {"id": "rain", "similarity_score": 0.2, "score": 1.0},
        {"id": "code", "similarity_score": 0.4, "score": 1.0},
    ]

    chroma_order = [item["id"] for item in rank_hybrid(fixture)]
    postgres_order = [item["id"] for item in rank_hybrid(fixture)]

    assert chroma_order == postgres_order == ["bank", "code", "rain"]


def test_hybrid_weights_are_validated():
    with pytest.raises(ValueError, match="negative"):
        hybrid_score({}, HybridWeights(semantic=-1))
    with pytest.raises(ValueError, match="positive"):
        hybrid_score({}, HybridWeights(0, 0, 0, 0))


def test_semantic_degraded_state_is_visible():
    class DegradedStore(MemoryStore):
        def load_all(self):
            return []

        def save_component(self, _component):
            pass

        def delete_component(self, _component_id):
            pass

        def get_component(self, _component_id):
            return None

        def search_similar(self, **_kwargs):
            return []

        def get_embedding_status(self):
            return {
                "available": False,
                "degraded": True,
                "reason": "model unavailable",
                "provider": {"model": "test"},
            }

    status = SemanticContextManager(memory_store=DegradedStore()).get_semantic_status()

    assert status["degraded"] is True
    assert status["reason"] == "model unavailable"


def test_semantic_manager_fallback_preserves_query_for_core_ranking():
    manager = SemanticContextManager()
    from ai_context_manager.components import TaskSummaryComponent

    manager.register_component(
        TaskSummaryComponent("relevant", "Python", "Fix the Python parser", score=0.4)
    )
    manager.register_component(
        TaskSummaryComponent("distractor", "Weather", "Rain forecast", score=1.0)
    )

    context = manager.get_semantic_context("Python parser", token_budget=100)

    assert "Fix the Python parser" in context
    assert context.index("Fix the Python parser") < context.index("Rain forecast")


def test_chroma_reembedding_updates_only_stale_records():
    provider = KeywordEmbeddingProvider("2")

    class FakeCollection:
        def __init__(self):
            current = embedding_metadata(provider, "weather")
            stale = embedding_metadata(KeywordEmbeddingProvider("1"), "finance")
            self.records = {
                "ids": ["stale", "current"],
                "metadatas": [stale, current],
                "documents": ["finance", "weather"],
            }
            self.upserts = []

        def get(self, include):
            assert include == ["metadatas", "documents"]
            return self.records

        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

    store = VectorMemoryStore.__new__(VectorMemoryStore)
    store.embedding_provider = provider
    store.last_embedding_error = None
    store.collection = FakeCollection()

    updated = store.reembed_all(stale_only=True)

    assert updated == 1
    assert store.collection.upserts[0]["ids"] == ["stale"]
    assert store.collection.upserts[0]["metadatas"][0]["embedding_version"] == "2"


def test_postgres_reembedding_updates_only_stale_records():
    provider = KeywordEmbeddingProvider("2")
    store = PostgreSQLVectorMemoryStore.__new__(PostgreSQLVectorMemoryStore)
    store.embedding_provider = provider
    store.last_embedding_error = None
    store.load_all = lambda: [
        {
            "id": "stale",
            "content": "finance",
            "metadata": embedding_metadata(
                KeywordEmbeddingProvider("1"),
                create_embedding_text({"content": "finance"}),
            ),
        },
        {
            "id": "current",
            "content": "weather",
            "metadata": embedding_metadata(
                provider, create_embedding_text({"content": "weather"})
            ),
        },
    ]
    saved = []
    store.save_component = saved.append

    updated = store.reembed_all(stale_only=True)

    assert updated == 1
    assert [component["id"] for component in saved] == ["stale"]


def test_embedding_failure_marks_store_degraded():
    class UnavailableProvider(KeywordEmbeddingProvider):
        def embed(self, _text):
            raise RuntimeError("provider offline")

    store = VectorMemoryStore.__new__(VectorMemoryStore)
    store.embedding_provider = UnavailableProvider()
    store.last_embedding_error = None

    with pytest.raises(RuntimeError, match="provider offline"):
        store._generate_embedding("query")

    status = store.get_embedding_status()
    assert status["degraded"] is True
    assert "provider offline" in status["reason"]
