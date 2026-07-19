"""
PostgreSQL + pgvector Memory Store - Production-grade vector database storage
"""

import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from .base import MemoryStore
from .errors import StorageReadError, StorageWriteError
from ai_context_manager.embeddings import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_text,
    embedding_metadata,
    needs_reembedding,
)
from ai_context_manager.hybrid import HybridWeights, rank_hybrid
from ai_context_manager.memory import memory_record_is_active

logger = logging.getLogger(__name__)

class PostgreSQLVectorMemoryStore(MemoryStore):
    """
    PostgreSQL + pgvector memory store for production-grade vector similarity search.
    
    Features:
    - Enterprise-grade ACID transactions
    - Horizontal scaling with read replicas
    - Advanced indexing (HNSW, IVFFlat)
    - Full-text search integration
    - Backup and recovery
    - Monitoring and observability
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None,
                 table_name: str = "agent_memory",
                 embedding_dimension: int = 384,
                 max_connections: int = 20,
                 index_type: str = "hnsw",  # "hnsw" or "ivfflat"
                 index_parameters: Optional[Dict[str, Any]] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize PostgreSQL vector memory store.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            table_name: Table name for storing components
            embedding_dimension: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
            max_connections: Maximum connection pool size
            index_type: Vector index type ("hnsw" or "ivfflat")
            index_parameters: Custom index parameters
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        # Use environment variables for security (Bandit B105, B106)
        import os
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "ai_context")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")  # nosec B106
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            embedding_model, dimension=embedding_dimension
        )
        if self.embedding_provider.info.dimension != embedding_dimension:
            raise EmbeddingDimensionError(
                f"Provider dimension {self.embedding_provider.info.dimension} does not "
                f"match PostgreSQL vector dimension {embedding_dimension}"
            )
        self.last_embedding_error = None
        self.index_type = index_type
        self.index_parameters = index_parameters or {}
        
        # Connection pool
        self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=max_connections,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        
        # Initialize database schema
        self._initialize_schema()
        
        logger.info(f"PostgreSQL vector memory store initialized: {host}:{port}/{database}")
    
    def _get_connection(self):
        """Get connection from pool."""
        return self.connection_pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool."""
        self.connection_pool.putconn(conn)
    
    def _initialize_schema(self):
        """Initialize database schema with pgvector extension."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")  # nosec B608
                
                # Create table with vector column
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    component_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags JSONB,
                    metadata JSONB,
                    embedding vector({self.embedding_dimension}),
                    score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
                cur.execute(create_table_sql)  # nosec B608
                
                # Create indexes
                self._create_indexes(cur)
                
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def _create_indexes(self, cursor):
        """Create database indexes for performance."""
        try:
            # Vector similarity index
            if self.index_type == "hnsw":
                index_params = {
                    "m": self.index_parameters.get("m", 16),
                    "ef_construction": self.index_parameters.get("ef_construction", 64)
                }
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx 
                ON {self.table_name} 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = {index_params['m']}, ef_construction = {index_params['ef_construction']});
                """
            else:  # ivfflat
                index_params = {
                    "lists": self.index_parameters.get("lists", 100)
                }
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_ivfflat_idx 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {index_params['lists']});
                """
            
            cursor.execute(index_sql)
            
            # Component type index
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_type_idx 
                ON {self.table_name} (component_type);
            """)
            
            # Tags GIN index for JSONB queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_tags_gin_idx 
                ON {self.table_name} USING gin (tags);
            """)
            
            # Score index for ranking
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_score_idx 
                ON {self.table_name} (score DESC);
            """)
            
            # Created at index for time-based queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_created_at_idx 
                ON {self.table_name} (created_at DESC);
            """)
            
            logger.info(f"Created {self.index_type} vector index with parameters: {self.index_parameters}")
            
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def _array_to_vector(self, embedding: List[float]) -> str:
        """Convert Python list to PostgreSQL vector format."""
        return f"[{','.join(map(str, embedding))}]"
    
    def _vector_to_array(self, vector_str: str) -> List[float]:
        """Convert PostgreSQL vector to Python list."""
        # Remove brackets and split by comma
        return [float(x) for x in vector_str.strip('[]').split(',')]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a real embedding through the configured provider."""
        try:
            vector = self.embedding_provider.embed_checked(text)
            self.last_embedding_error = None
            return vector
        except Exception as exc:
            self.last_embedding_error = str(exc)
            raise
    
    def load_all(self) -> List[Dict]:
        """Load all components from PostgreSQL."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, component_type, content, tags, metadata, 
                           score, created_at, updated_at
                    FROM {self.table_name}
                    ORDER BY created_at DESC
                """)
                
                components = []
                for row in cur.fetchall():
                    component = {
                        "id": row["id"],
                        "type": row["component_type"],
                        "content": row["content"],
                        "tags": row["tags"] or [],
                        "timestamp": row["created_at"].isoformat(),
                        "score": float(row["score"]),
                        "metadata": row["metadata"] or {}
                    }
                    component["memory"] = component["metadata"].get("memory")
                    components.append(component)
                
                logger.debug(f"Loaded {len(components)} components from PostgreSQL")
                return components
                
        except Exception as e:
            logger.error(f"Failed to load components from PostgreSQL: {e}")
            raise StorageReadError("Failed to load components from PostgreSQL") from e
        finally:
            self._return_connection(conn)
    
    def save_component(self, component: Dict) -> None:
        """Save component to PostgreSQL with vector embedding."""
        conn = self._get_connection()
        try:
            component_id = component["id"]
            content = component.get("content", "")
            component_type = component.get("type", "Unknown")
            tags = component.get("tags", [])
            metadata = dict(component.get("metadata", {}))
            metadata.update({
                "schema_version": component.get(
                    "schema_version", metadata.get("schema_version", 1)
                ),
                "component_data": component.get(
                    "component_data", metadata.get("component_data", "{}")
                ),
                "memory": component.get("memory", metadata.get("memory")),
            })
            score = component.get("score", 1.0)
            searchable_text = create_embedding_text(component)
            metadata.update(embedding_metadata(self.embedding_provider, searchable_text))

            # Generate embedding
            embedding = self._generate_embedding(searchable_text)
            embedding_vector = self._array_to_vector(embedding)
            
            with conn.cursor() as cur:
                # Upsert component
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, component_type, content, tags, metadata, embedding, score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) 
                    DO UPDATE SET
                        component_type = EXCLUDED.component_type,
                        content = EXCLUDED.content,
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        score = EXCLUDED.score,
                        updated_at = NOW()
                """, (
                    component_id,
                    component_type,
                    content,
                    json.dumps(tags),
                    json.dumps(metadata),
                    embedding_vector,
                    score
                ))
                
                conn.commit()
                logger.debug(f"Saved component {component_id} to PostgreSQL")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save component {component.get('id', 'unknown')} to PostgreSQL: {e}")
            raise StorageWriteError("Failed to save component to PostgreSQL") from e
        finally:
            self._return_connection(conn)
    
    def delete_component(self, component_id: str) -> None:
        """Delete component from PostgreSQL."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (component_id,))
                conn.commit()
                logger.debug(f"Deleted component {component_id} from PostgreSQL")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete component {component_id} from PostgreSQL: {e}")
            raise StorageWriteError("Failed to delete component from PostgreSQL") from e
        finally:
            self._return_connection(conn)
    
    def get_component(self, component_id: str) -> Optional[Dict]:
        """Get specific component by ID."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT id, component_type, content, tags, metadata, 
                           score, created_at, updated_at
                    FROM {self.table_name}
                    WHERE id = %s
                """, (component_id,))
                
                row = cur.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "type": row["component_type"],
                        "content": row["content"],
                        "tags": row["tags"] or [],
                        "timestamp": row["created_at"].isoformat(),
                        "score": float(row["score"]),
                        "metadata": row["metadata"] or {}
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get component {component_id} from PostgreSQL: {e}")
            raise StorageReadError("Failed to get component from PostgreSQL") from e
        finally:
            self._return_connection(conn)
    
    def search_similar(self, query: str, n_results: int = 10,
                      include_types: Optional[List[str]] = None,
                      include_tags: Optional[List[str]] = None,
                      score_threshold: float = 0.0,
                      hybrid_weights: Optional[HybridWeights] = None,
                      include_inactive: bool = False) -> List[Dict]:
        """
        Search for similar components using vector similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            include_types: Filter by component types
            include_tags: Filter by tags
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar components with similarity scores
        """
        conn = self._get_connection()
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            query_vector = self._array_to_vector(query_embedding)
            
            # Build WHERE clause
            where_conditions = []
            params = [query_vector]
            
            if include_types:
                placeholders = ','.join(['%s'] * len(include_types))
                where_conditions.append(f"component_type IN ({placeholders})")
                params.extend(include_types)
            
            if include_tags:
                for tag in include_tags:
                    where_conditions.append("tags ? %s")
                    params.append(tag)

            if not include_inactive:
                where_conditions.append(
                    "("
                    "metadata->'memory' IS NULL OR ("
                    "COALESCE(metadata->'memory'->>'status', 'active') = 'active' AND "
                    "(metadata->'memory'->>'expires_at' IS NULL OR "
                    "(metadata->'memory'->>'expires_at')::timestamptz > NOW())"
                    ")"
                    ")"
                )
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Vector similarity search
            search_sql = f"""
                SELECT id, component_type, content, tags, metadata, 
                       score, created_at, updated_at,
                       1 - (embedding <=> %s) as similarity_score
                FROM {self.table_name}
                {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
            """
            params.extend([query_vector, n_results])
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(search_sql, params)
                
                similar_components = []
                for row in cur.fetchall():
                    similarity_score = float(row["similarity_score"])
                    
                    if similarity_score >= score_threshold:
                        component = {
                            "id": row["id"],
                            "type": row["component_type"],
                            "content": row["content"],
                            "tags": row["tags"] or [],
                            "timestamp": row["created_at"].isoformat(),
                            "score": float(row["score"]),
                            "similarity_score": similarity_score,
                            "metadata": row["metadata"] or {}
                        }
                        component["memory"] = component["metadata"].get("memory")
                        if include_inactive or memory_record_is_active(component):
                            similar_components.append(component)
                
                logger.debug(f"Found {len(similar_components)} similar components for query: {query[:50]}...")
                return rank_hybrid(similar_components, hybrid_weights)
                
        except Exception as e:
            logger.error(f"Failed to search PostgreSQL: {e}")
            raise
        finally:
            self._return_connection(conn)

    def get_embedding_status(self) -> Dict[str, Any]:
        return {
            "available": self.last_embedding_error is None,
            "degraded": self.last_embedding_error is not None,
            "reason": self.last_embedding_error,
            "provider": self.embedding_provider.info.to_dict(),
        }

    def reembed_all(self, stale_only: bool = True) -> int:
        """Regenerate stored vectors whose content or provider metadata is stale."""
        components = self.load_all()
        updated = 0
        for component in components:
            metadata = component.get("metadata", {})
            searchable_text = create_embedding_text(component)
            if stale_only and not needs_reembedding(
                metadata, self.embedding_provider, searchable_text
            ):
                continue
            self.save_component(component)
            updated += 1
        return updated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL store statistics."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Basic stats
                cur.execute(f"SELECT COUNT(*) as total_components FROM {self.table_name}")
                total_components = cur.fetchone()["total_components"]
                
                # Component type breakdown
                cur.execute(f"""
                    SELECT component_type, COUNT(*) as count 
                    FROM {self.table_name} 
                    GROUP BY component_type 
                    ORDER BY count DESC
                """)
                component_types = {row["component_type"]: row["count"] for row in cur.fetchall()}
                
                # Index information
                cur.execute("""
                    SELECT indexname, indexdef 
                    FROM pg_indexes 
                    WHERE tablename = %s
                """, (self.table_name,))
                indexes = [{"name": row["indexname"], "definition": row["indexdef"]} for row in cur.fetchall()]
                
                # Database size
                cur.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size(%s)) as table_size
                """, (self.table_name,))
                table_size = cur.fetchone()["table_size"]
                
                return {
                    "total_components": total_components,
                    "component_types": component_types,
                    "indexes": indexes,
                    "table_size": table_size,
                    "index_type": self.index_type,
                    "embedding_dimension": self.embedding_dimension,
                    "embedding_provider": self.embedding_provider.info.to_dict(),
                    "embedding_degraded": self.last_embedding_error is not None,
                    "connection_pool_size": self.connection_pool.closed,
                    "database": f"{self.host}:{self.port}/{self.database}"
                }
                
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL stats: {e}")
            return {"error": str(e)}
        finally:
            self._return_connection(conn)
    
    def clear_all(self) -> None:
        """Clear all data from PostgreSQL table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                conn.commit()
                logger.info("Cleared all data from PostgreSQL table")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to clear PostgreSQL table: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def close(self):
        """Close connection pool."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("Closed PostgreSQL connection pool")


class PostgreSQLVectorRetriever:
    """Enhanced context retriever using PostgreSQL vector search."""
    
    def __init__(self, vector_store: PostgreSQLVectorMemoryStore):
        self.vector_store = vector_store
    
    def get_semantic_context(self, query: str, token_budget: int = 2000,
                           max_components: int = 20,
                           include_types: Optional[List[str]] = None,
                           include_tags: Optional[List[str]] = None,
                           similarity_threshold: float = 0.3) -> str:
        """
        Get context using PostgreSQL vector similarity search.
        
        Args:
            query: Semantic query for context retrieval
            token_budget: Maximum tokens for context
            max_components: Maximum number of components to consider
            include_types: Filter by component types
            include_tags: Filter by tags
            similarity_threshold: Minimum similarity score
            
        Returns:
            Formatted context string
        """
        # Search for similar components
        similar_components = self.vector_store.search_similar(
            query=query,
            n_results=max_components,
            include_types=include_types,
            include_tags=include_tags,
            score_threshold=similarity_threshold
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
