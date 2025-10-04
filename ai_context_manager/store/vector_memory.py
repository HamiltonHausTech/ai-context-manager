"""
Vector Database Memory Store - Semantic similarity-based memory storage
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import sys
    # Check Python version compatibility for sentence_transformers
    if sys.version_info >= (3, 10):
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    else:
        # Python 3.9 has compatibility issues with newer sentence_transformers
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        SentenceTransformer = None
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from .base import MemoryStore

logger = logging.getLogger(__name__)

class VectorMemoryStore(MemoryStore):
    """
    Vector database memory store using ChromaDB for semantic similarity search.
    Provides much more efficient retrieval for agent context management.
    """
    
    def __init__(self, collection_name: str = "agent_memory", 
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector memory store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            import sys
            if sys.version_info < (3, 10):
                raise ImportError("SentenceTransformers requires Python 3.10+ for compatibility. Current version: {}.{}.{}".format(
                    sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
            else:
                raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Vector memory store initialized with {self.collection.count()} items")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def _create_document_text(self, component: Dict[str, Any]) -> str:
        """Create searchable text from component data."""
        content = component.get("content", "")
        component_type = component.get("type", "Unknown")
        tags = component.get("tags", [])
        
        # Create rich searchable text
        searchable_text = f"{content}\n"
        searchable_text += f"Type: {component_type}\n"
        if tags:
            searchable_text += f"Tags: {', '.join(tags)}\n"
        
        # Add metadata for better search
        if component_type == "AgentGoalComponent":
            searchable_text += f"Goal: {content}\n"
        elif component_type == "TaskSummaryComponent":
            searchable_text += f"Task: {component.get('task_name', 'Unknown')}\n"
        elif component_type == "LongTermMemoryComponent":
            searchable_text += f"Learning: {content}\n"
        
        return searchable_text
    
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
                        "score": metadata.get("score", 1.0)
                    }
                    components.append(component)
            
            logger.debug(f"Loaded {len(components)} components from vector store")
            return components
            
        except Exception as e:
            logger.error(f"Failed to load components from vector store: {e}")
            return []
    
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
                "created_at": datetime.utcnow().isoformat()
            }
            
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
    
    def delete_component(self, component_id: str) -> None:
        """Delete component from vector store."""
        try:
            self.collection.delete(ids=[component_id])
            logger.debug(f"Deleted component {component_id} from vector store")
        except Exception as e:
            logger.error(f"Failed to delete component {component_id} from vector store: {e}")
    
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
                    "score": metadata.get("score", 1.0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get component {component_id} from vector store: {e}")
            return None
    
    def search_similar(self, query: str, n_results: int = 10, 
                      include_types: Optional[List[str]] = None,
                      include_tags: Optional[List[str]] = None) -> List[Dict]:
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
            if include_tags:
                where_clause["tags"] = {"$in": include_tags}
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
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
                    similar_components.append(component)
            
            logger.debug(f"Found {len(similar_components)} similar components for query: {query[:50]}...")
            return similar_components
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            count = self.collection.count()
            return {
                "total_components": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
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
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared all data from vector store")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")


class SemanticContextRetriever:
    """
    Enhanced context retriever using semantic similarity search.
    """
    
    def __init__(self, vector_store: VectorMemoryStore):
        self.vector_store = vector_store
    
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
        
        # Build context with token budget management
        context_parts = []
        used_tokens = 0
        
        for component in similar_components:
            content = component["content"]
            similarity_score = component.get("similarity_score", 0.0)
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(content.split()) * 1.3
            
            if used_tokens + estimated_tokens <= token_budget:
                context_parts.append(f"[{component['id']}] {component['type']} (similarity: {similarity_score:.2f})\n{content}")
                used_tokens += estimated_tokens
            else:
                # Try to fit a summary if we're close to budget
                remaining_tokens = token_budget - used_tokens
                if remaining_tokens > 50:  # Minimum viable summary
                    summary = content[:remaining_tokens * 4]  # Rough character estimation
                    context_parts.append(f"[{component['id']}] {component['type']} (similarity: {similarity_score:.2f}) [SUMMARY]\n{summary}...")
                break
        
        return "\n\n".join(context_parts)
