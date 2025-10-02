#!/usr/bin/env python3
"""
Test script for vector database functionality.
Demonstrates semantic similarity search and retrieval.
"""

import logging
from ai_context_manager.semantic_context_manager import SemanticContextManager
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config, load_summarizer
from ai_context_manager.feedback import Feedback

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_vector_database():
    """Test vector database functionality."""
    print("=" * 60)
    print("Vector Database Memory Store Test")
    print("=" * 60)
    
    try:
        # Load configuration
        config = Config("config.toml")
        config_data = config.data
        feedback_store, memory_store = load_stores_from_config(config_data)
        
        # Create semantic context manager
        feedback = Feedback(store=feedback_store)
        ctx = SemanticContextManager(
            feedback=feedback,
            memory_store=memory_store,
            summarizer=load_summarizer(config),
            config=config_data
        )
        
        print(f"[STATUS] Memory store type: {type(memory_store).__name__}")
        print(f"[STATUS] Semantic search enabled: {ctx.semantic_retriever is not None}")
        
        # Add some test components
        print("\n[SETUP] Adding test components...")
        
        from ai_context_manager.components.task_summary import TaskSummaryComponent
        from ai_context_manager.components.longterm_summary import LongTermMemoryComponent
        
        # Add task components
        tasks = [
            TaskSummaryComponent(
                id="task-market-research",
                task_name="Market Research Analysis",
                summary="Analyzed AI market trends showing 40% growth in context management systems",
                tags=["research", "market", "ai", "growth"]
            ),
            TaskSummaryComponent(
                id="task-competitor-analysis",
                task_name="Competitor Analysis",
                summary="Studied LangChain's context management approach and identified key advantages",
                tags=["competitor", "analysis", "langchain", "context"]
            ),
            TaskSummaryComponent(
                id="task-user-feedback",
                task_name="User Feedback Collection",
                summary="Collected feedback from users about context management preferences and pain points",
                tags=["user", "feedback", "preferences", "pain-points"]
            )
        ]
        
        # Add learning components
        learnings = [
            LongTermMemoryComponent(
                id="learning-privacy-trend",
                content="Privacy-focused AI systems are gaining 300% more adoption than cloud-based alternatives",
                source="market-research",
                timestamp="2025-01-15T10:00:00Z",
                tags=["privacy", "ai", "adoption", "trend"]
            ),
            LongTermMemoryComponent(
                id="learning-vector-efficiency",
                content="Vector databases provide 10x faster semantic search compared to traditional keyword matching",
                source="performance-testing",
                timestamp="2025-01-15T11:00:00Z",
                tags=["vector", "database", "performance", "semantic"]
            ),
            LongTermMemoryComponent(
                id="learning-agent-memory",
                content="Agent memory systems are becoming a distinct category in AI infrastructure funding",
                source="investment-research",
                timestamp="2025-01-15T12:00:00Z",
                tags=["agent", "memory", "funding", "infrastructure"]
            )
        ]
        
        # Register all components
        for task in tasks:
            ctx.register_component(task)
        
        for learning in learnings:
            ctx.register_component(learning)
        
        print(f"[SETUP] Added {len(tasks)} tasks and {len(learnings)} learnings")
        
        # Test semantic search
        print("\n[TEST] Semantic similarity search...")
        
        test_queries = [
            "AI market growth trends",
            "privacy concerns in artificial intelligence",
            "vector database performance benefits",
            "user experience feedback collection",
            "competitor analysis methods"
        ]
        
        for query in test_queries:
            print(f"\n[QUERY] '{query}'")
            
            # Search for similar components
            similar = ctx.search_similar_components(query, n_results=3)
            
            for i, result in enumerate(similar, 1):
                similarity_score = result.get("similarity_score", 0.0)
                component_type = result.get("type", "Unknown")
                content_preview = result.get("content", "")[:100]
                
                print(f"  {i}. [{component_type}] (similarity: {similarity_score:.3f})")
                print(f"     {content_preview}...")
        
        # Test semantic context retrieval
        print("\n[TEST] Semantic context retrieval...")
        
        context_queries = [
            "What are the latest trends in AI market research?",
            "How do privacy concerns affect AI adoption?",
            "What are the benefits of vector databases for semantic search?"
        ]
        
        for query in context_queries:
            print(f"\n[CONTEXT QUERY] '{query}'")
            context = ctx.get_semantic_context(query, token_budget=500)
            print(f"[RESULT] {len(context)} characters retrieved")
            print(f"[PREVIEW] {context[:200]}...")
        
        # Test memory statistics
        print("\n[STATS] Memory statistics:")
        stats = ctx.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test traditional vs semantic comparison
        print("\n[COMPARISON] Traditional vs Semantic retrieval...")
        
        test_query = "AI market research trends"
        
        # Traditional retrieval
        traditional_context = ctx.get_context(
            include_tags=["research", "market"],
            token_budget=500,
            summarize_if_needed=True
        )
        
        # Semantic retrieval
        semantic_context = ctx.get_semantic_context(test_query, token_budget=500)
        
        print(f"[TRADITIONAL] Retrieved {len(traditional_context)} characters")
        print(f"[SEMANTIC] Retrieved {len(semantic_context)} characters")
        
        print("\n[SUCCESS] Vector database test completed successfully!")
        
    except ImportError as e:
        print(f"[ERROR] Vector database dependencies not available: {e}")
        print("[INFO] Install with: pip install chromadb sentence-transformers")
        print("[INFO] Or use JSON storage by setting type = 'json' in config.toml")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_database()
