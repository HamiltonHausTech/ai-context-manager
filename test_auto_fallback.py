#!/usr/bin/env python3
"""
Test script to demonstrate auto-fallback summarizer functionality.
This script will show how the system automatically detects Ollama availability
and falls back gracefully when it's not available.
"""

import logging
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config, load_summarizer
from ai_context_manager.components.task_summary import TaskSummaryComponent

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def test_auto_fallback():
    """Test the auto-fallback summarizer functionality."""
    print("=" * 60)
    print("Testing Auto-Fallback Summarizer")
    print("=" * 60)
    
    try:
        # Load configuration
        config_obj = Config("config.toml")
        config = config_obj.data
        feedback_store, memory_store = load_stores_from_config(config)
        
        # Initialize context manager with auto-fallback summarizer
        ctx = ContextManager(
            memory_store=memory_store,
            summarizer=load_summarizer(config_obj)
        )
        
        # Get summarizer status
        if hasattr(ctx.summarizer, 'get_status'):
            status = ctx.summarizer.get_status()
            print(f"\n[STATUS] Summarizer Status:")
            print(f"  Type: {status['type']}")
            print(f"  Ollama Available: {status['ollama_available']}")
            print(f"  Ollama Host: {status['ollama_host']}")
            print(f"  Fallback: {status['fallback_summarizer']}")
        
        # Create a test component with long content
        long_text = """
        This is a very long task description that will definitely exceed our token budget.
        It contains multiple paragraphs of detailed information about a complex software project.
        
        The project involves building a comprehensive AI context management system with the following features:
        1. Modular architecture with pluggable components
        2. Multiple storage backends (JSON and SQLite)
        3. Intelligent summarization with fallback mechanisms
        4. Feedback learning system for component prioritization
        5. Token-aware context budgeting
        6. Network-aware configuration management
        
        This content is intentionally verbose to test the summarization capabilities.
        When the token budget is exceeded, the system should automatically summarize
        this content to fit within the specified limits.
        """ * 3  # Make it even longer
        
        task = TaskSummaryComponent(
            id="test-auto-fallback",
            task_name="Auto-Fallback Test Task",
            summary=long_text,
            tags=["test", "auto-fallback", "summarization"]
        )
        
        ctx.register_component(task)
        
        # Test context generation with small token budget to force summarization
        print(f"\n[TEST] Getting context with small token budget (100 tokens)...")
        context = ctx.get_context(
            include_tags=["test"],
            token_budget=100,
            summarize_if_needed=True,
            dry_run=True
        )
        
        print(f"\n[TEST] Getting actual context...")
        context = ctx.get_context(
            include_tags=["test"],
            token_budget=100,
            summarize_if_needed=True
        )
        
        print(f"\n[RESULT] Context length: {len(context)} characters")
        print(f"[RESULT] Context preview: {context[:200]}...")
        
        # Test status again to see if anything changed
        if hasattr(ctx.summarizer, 'get_status'):
            status = ctx.summarizer.get_status()
            print(f"\n[STATUS] After testing:")
            print(f"  Ollama Available: {status['ollama_available']}")
            print(f"  Last Health Check: {status['last_health_check']}")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auto_fallback()
