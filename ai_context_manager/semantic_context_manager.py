"""
Semantic Context Manager - Enhanced context management with vector database support
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from .context_manager import ContextManager
from .store.vector_memory import VectorMemoryStore, SemanticContextRetriever
from .components import ContextComponent

logger = logging.getLogger(__name__)

class SemanticContextManager(ContextManager):
    """
    Enhanced context manager with semantic similarity search capabilities.
    Falls back to traditional tag-based retrieval when vector database is not available.
    """
    
    def __init__(self, feedback=None, memory_store=None, config=None, summarizer=None):
        super().__init__(feedback, memory_store, config, summarizer)
        
        # Initialize semantic retriever if vector store is available
        self.semantic_retriever = None
        if isinstance(memory_store, VectorMemoryStore):
            self.semantic_retriever = SemanticContextRetriever(memory_store)
            logger.info("Semantic context retrieval enabled")
        else:
            logger.info("Using traditional tag-based context retrieval")
    
    def get_semantic_context(self, query: str, token_budget: int = 2000,
                           max_components: int = 20,
                           include_types: Optional[List[str]] = None,
                           include_tags: Optional[List[str]] = None) -> str:
        """
        Get context using semantic similarity search.
        Falls back to traditional retrieval if vector database is not available.
        
        Args:
            query: Semantic query for context retrieval
            token_budget: Maximum tokens for context
            max_components: Maximum number of components to consider
            include_types: Filter by component types
            include_tags: Filter by tags
            
        Returns:
            Formatted context string
        """
        if self.semantic_retriever:
            # Use semantic search
            return self.semantic_retriever.get_semantic_context(
                query=query,
                token_budget=token_budget,
                max_components=max_components,
                include_types=include_types,
                include_tags=include_tags
            )
        else:
            # Fall back to traditional tag-based retrieval
            logger.debug("Falling back to traditional context retrieval")
            return self.get_context(
                include_tags=include_tags,
                component_types=include_types,
                token_budget=token_budget,
                summarize_if_needed=True
            )
    
    def get_agent_semantic_context(self, agent_id: str, query: str, 
                                 token_budget: int = 2000) -> str:
        """
        Get semantic context for a specific agent.
        Combines semantic search with agent-specific filtering.
        """
        # Add agent-specific tags to the query
        agent_query = f"{query} agent:{agent_id}"
        
        # Filter for agent-specific components
        agent_tags = [agent_id, "agent"]
        
        return self.get_semantic_context(
            query=agent_query,
            token_budget=token_budget,
            include_tags=agent_tags
        )
    
    def search_similar_components(self, query: str, n_results: int = 10,
                                include_types: Optional[List[str]] = None,
                                include_tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for similar components using semantic similarity.
        Returns components with similarity scores.
        """
        if isinstance(self.memory_store, VectorMemoryStore):
            return self.memory_store.search_similar(
                query=query,
                n_results=n_results,
                include_types=include_types,
                include_tags=include_tags
            )
        else:
            # Fall back to traditional search
            logger.warning("Semantic search not available, using traditional search")
            components = list(self.components.values())
            
            if include_tags:
                components = [c for c in components if c.matches_tags(include_tags)]
            if include_types:
                components = [c for c in components if c.__class__.__name__ in include_types]
            
            # Return as list of dicts with similarity scores
            results = []
            for comp in components[:n_results]:
                results.append({
                    "id": comp.id,
                    "type": comp.__class__.__name__,
                    "content": comp.get_content(),
                    "tags": comp.tags,
                    "similarity_score": 1.0,  # Default score for traditional search
                    "component": comp
                })
            
            return results
    
    def get_context_with_semantic_fallback(self, query: Optional[str] = None,
                                         include_tags: Optional[List[str]] = None,
                                         component_types: Optional[List[str]] = None,
                                         token_budget: Optional[int] = None,
                                         semantic_search: bool = True) -> str:
        """
        Get context with intelligent fallback between semantic and traditional search.
        
        Args:
            query: Semantic query (if provided, will use semantic search)
            include_tags: Traditional tag filtering
            component_types: Component type filtering
            token_budget: Token budget limit
            semantic_search: Whether to prefer semantic search when available
            
        Returns:
            Context string
        """
        # If semantic query is provided and semantic search is available, use it
        if query and semantic_search and self.semantic_retriever:
            logger.debug(f"Using semantic search for query: {query[:50]}...")
            return self.get_semantic_context(
                query=query,
                token_budget=token_budget or 2000,
                include_types=component_types,
                include_tags=include_tags
            )
        else:
            # Use traditional retrieval
            logger.debug("Using traditional context retrieval")
            return self.get_context(
                include_tags=include_tags,
                component_types=component_types,
                token_budget=token_budget,
                summarize_if_needed=True
            )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "total_components": len(self.components),
            "semantic_search_enabled": self.semantic_retriever is not None,
            "memory_store_type": type(self.memory_store).__name__ if self.memory_store else "None"
        }
        
        # Add vector store specific stats
        if isinstance(self.memory_store, VectorMemoryStore):
            vector_stats = self.memory_store.get_stats()
            stats.update(vector_stats)
        
        # Add component type breakdown
        component_types = {}
        for comp in self.components.values():
            comp_type = comp.__class__.__name__
            component_types[comp_type] = component_types.get(comp_type, 0) + 1
        stats["component_types"] = component_types
        
        return stats
