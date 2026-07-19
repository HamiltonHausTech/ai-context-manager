"""
Vector Database Memory Store - Semantic similarity-based memory storage
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .base import MemoryStore
from .errors import StorageReadError, StorageWriteError
from ai_context_manager.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_text,
    embedding_metadata,
    needs_reembedding,
)
from ai_context_manager.hybrid import HybridWeights, rank_hybrid
from ai_context_manager.tokenization import estimate_tokens, truncate_to_token_budget
from ai_context_manager.memory import memory_record_is_active

logger = logging.getLogger(__name__)

class VectorMemoryStore(MemoryStore):
    """
    Vector database memory store using ChromaDB for semantic similarity search.
    Provides much more efficient retrieval for agent context management.
    """
    
    def __init__(self, collection_name: str = "agent_memory",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize vector memory store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            embedding_model
        )
        self.last_embedding_error = None
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_identifier": self.embedding_provider.info.identifier,
            }
        )
        
        logger.info(f"Vector memory store initialized with {self.collection.count()} items")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embedding = self.embedding_provider.embed_checked(text)
            self.last_embedding_error = None
            return embedding
        except Exception as e:
            self.last_embedding_error = str(e)
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _create_document_text(self, component: Dict[str, Any]) -> str:
        """Create searchable text from component data."""
        return create_embedding_text(component)
    
    def load_all(self) -> List[Dict]:
        """Load all components from vector store."""
        try:
            results = self.collection.get(include=["metadatas"])
            components = []
            
            for i, (doc_id, metadata) in enumerate(zip(results["ids"], results["metadatas"])):
                if metadata:
                    component = {
                        "id": metadata.get("component_id", doc_id),
                        "type": metadata.get("type", "Unknown"),
                        "content": metadata.get("content", ""),
                        "tags": json.loads(metadata.get("tags", "[]")),
                        "timestamp": metadata.get("timestamp", ""),
                        "score": metadata.get("score", 1.0),
                        "schema_version": metadata.get("schema_version"),
                        "component_data": metadata.get("component_data"),
                        "memory": json.loads(metadata.get("memory", "null")),
                    }
                    components.append(component)
            
            logger.debug(f"Loaded {len(components)} components from vector store")
            return components
            
        except Exception as e:
            logger.error(f"Failed to load components from vector store: {e}")
            raise StorageReadError("Failed to load components from vector store") from e
    
    def save_component(self, component: Dict) -> None:
        """Save component to vector store with embedding."""
        try:
            component_id = component["id"]
            content = component.get("content", "")
            component_type = component.get("type", "Unknown")
            tags = component.get("tags", [])
            
            # Create searchable text
            searchable_text = self._create_document_text(component)
            
            # Generate embedding
            embedding = self._generate_embedding(searchable_text)
            
            # Prepare metadata
            metadata = {
                "component_id": component_id,
                "type": component_type,
                "content": content,
                "tags": json.dumps(tags),
                "timestamp": component.get("timestamp", datetime.utcnow().isoformat()),
                "score": component.get("score", 1.0),
                "schema_version": component.get("schema_version", 1),
                "component_data": component.get("component_data", "{}"),
                "memory": json.dumps(component.get("memory")),
                "created_at": datetime.utcnow().isoformat()
            }
            metadata.update(embedding_metadata(self.embedding_provider, searchable_text))
            
            # Save to vector store
            self.collection.upsert(
                ids=[component_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[searchable_text]
            )
            
            logger.debug(f"Saved component {component_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to save component {component.get('id', 'unknown')} to vector store: {e}")
            raise StorageWriteError("Failed to save component to vector store") from e
    
    def delete_component(self, component_id: str) -> None:
        """Delete component from vector store."""
        try:
            self.collection.delete(ids=[component_id])
            logger.debug(f"Deleted component {component_id} from vector store")
        except Exception as e:
            logger.error(f"Failed to delete component {component_id} from vector store: {e}")
            raise StorageWriteError("Failed to delete component from vector store") from e
    
    def get_component(self, component_id: str) -> Optional[Dict]:
        """Get specific component by ID."""
        try:
            results = self.collection.get(
                ids=[component_id],
                include=["metadatas"]
            )
            
            if results["ids"] and results["metadatas"]:
                metadata = results["metadatas"][0]
                return {
                    "id": metadata.get("component_id", component_id),
                    "type": metadata.get("type", "Unknown"),
                    "content": metadata.get("content", ""),
                    "tags": json.loads(metadata.get("tags", "[]")),
                    "timestamp": metadata.get("timestamp", ""),
                    "score": metadata.get("score", 1.0),
                    "schema_version": metadata.get("schema_version"),
                    "component_data": metadata.get("component_data"),
                    "memory": json.loads(metadata.get("memory", "null")),
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get component {component_id} from vector store: {e}")
            raise StorageReadError("Failed to get component from vector store") from e
    
    def search_similar(self, query: str, n_results: int = 10,
                      include_types: Optional[List[str]] = None,
                      include_tags: Optional[List[str]] = None,
                      hybrid_weights: Optional[HybridWeights] = None,
                      include_inactive: bool = False) -> List[Dict]:
        """
        Search for similar components using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            include_types: Filter by component types
            include_tags: Filter by tags
            
        Returns:
            List of similar components with similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            
            # Build where clause for filtering
            where_clause = {}
            if include_types:
                where_clause["type"] = {"$in": include_types}
            fetch_count = max(n_results, self.collection.count())

            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_count,
                where=where_clause if where_clause else None,
                include=["metadatas", "distances", "documents"]
            )
            
            # Format results
            similar_components = []
            if results["ids"] and results["metadatas"]:
                for i, (doc_id, metadata, distance) in enumerate(zip(
                    results["ids"][0], 
                    results["metadatas"][0], 
                    results["distances"][0]
                )):
                    component = {
                        "id": metadata.get("component_id", doc_id),
                        "type": metadata.get("type", "Unknown"),
                        "content": metadata.get("content", ""),
                        "tags": json.loads(metadata.get("tags", "[]")),
                        "timestamp": metadata.get("timestamp", ""),
                        "score": metadata.get("score", 1.0),
                        "similarity_score": 1.0 - distance,  # Convert distance to similarity
                        "distance": distance
                    }
                    component["memory"] = json.loads(metadata.get("memory", "null"))
                    tags_match = not include_tags or all(
                        tag in component["tags"] for tag in include_tags
                    )
                    if tags_match and (
                        include_inactive or memory_record_is_active(component)
                    ):
                        similar_components.append(component)
            
            logger.debug(f"Found {len(similar_components)} similar components for query: {query[:50]}...")
            return rank_hybrid(similar_components, hybrid_weights)[:n_results]
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise

    def get_embedding_status(self) -> Dict[str, Any]:
        return {
            "available": self.last_embedding_error is None,
            "degraded": self.last_embedding_error is not None,
            "reason": self.last_embedding_error,
            "provider": self.embedding_provider.info.to_dict(),
        }

    def reembed_all(self, stale_only: bool = True) -> int:
        """Regenerate vectors after provider, model, version, or content changes."""
        try:
            results = self.collection.get(include=["metadatas", "documents"])
            updated = 0
            for component_id, metadata, document in zip(
                results["ids"], results["metadatas"], results["documents"]
            ):
                metadata = dict(metadata or {})
                if stale_only and not needs_reembedding(
                    metadata, self.embedding_provider, document
                ):
                    continue
                metadata.update(embedding_metadata(self.embedding_provider, document))
                self.collection.upsert(
                    ids=[component_id],
                    embeddings=[self._generate_embedding(document)],
                    metadatas=[metadata],
                    documents=[document],
                )
                updated += 1
            return updated
        except Exception as exc:
            raise StorageWriteError("Failed to re-embed vector store") from exc
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_components": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "embedding_provider": self.embedding_provider.info.to_dict(),
                "embedding_degraded": self.last_embedding_error is not None,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"error": str(e)}
    
    def clear_all(self) -> None:
        """Clear all data from vector store."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_identifier": self.embedding_provider.info.identifier,
                }
            )
            logger.info("Cleared all data from vector store")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")


class SemanticContextRetriever:
    """
    Enhanced context retriever using semantic similarity search.
    """
    
    def __init__(self, vector_store: VectorMemoryStore, feedback=None):
        self.vector_store = vector_store
        self.feedback = feedback
    
    def get_semantic_context(self, query: str, token_budget: int = 2000,
                           max_components: int = 20,
                           include_types: Optional[List[str]] = None,
                           include_tags: Optional[List[str]] = None) -> str:
        """
        Get context using semantic similarity search.
        
        Args:
            query: Semantic query for context retrieval
            token_budget: Maximum tokens for context
            max_components: Maximum number of components to consider
            include_types: Filter by component types
            include_tags: Filter by tags
            
        Returns:
            Formatted context string
        """
        # Search for similar components
        similar_components = self.vector_store.search_similar(
            query=query,
            n_results=max_components,
            include_types=include_types,
            include_tags=include_tags
        )
        if self.feedback:
            for component in similar_components:
                component["feedback_score"] = (
                    self.feedback.get_average_score(component["id"]) * 0.7
                    + self.feedback.get_average_score_by_type(component["type"]) * 0.3
                )
            similar_components = rank_hybrid(similar_components)
        
        # Build context with token budget management
        context_parts = []
        used_tokens = 0
        
        for component in similar_components:
            content = component["content"]
            similarity_score = component.get("similarity_score", 0.0)
            
            estimated_tokens = estimate_tokens(content)
            
            if used_tokens + estimated_tokens <= token_budget:
                context_parts.append(f"[{component['id']}] {component['type']} (similarity: {similarity_score:.2f})\n{content}")
                used_tokens += estimated_tokens
            else:
                # Try to fit a summary if we're close to budget
                remaining_tokens = token_budget - used_tokens
                if remaining_tokens > 50:  # Minimum viable summary
                    summary = truncate_to_token_budget(content, int(remaining_tokens))
                    context_parts.append(f"[{component['id']}] {component['type']} (similarity: {similarity_score:.2f}) [SUMMARY]\n{summary}...")
                break
        
        return "\n\n".join(context_parts)
