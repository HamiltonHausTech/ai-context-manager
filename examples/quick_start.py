#!/usr/bin/env python3
"""
Quick Start Examples for AI Context Manager
"""

from ai_context_manager.simple_api import (
    create_agent_context_manager, 
    create_context_manager,
    quick_setup
)

def example_1_simple_task_manager():
    """Example 1: Simple task management."""
    print("[EXAMPLE 1] Simple Task Manager")
    print("=" * 40)
    
    # Create a simple context manager
    ctx = create_context_manager()
    
    # Add some tasks
    ctx.add_task("task-1", "Market Research", "Analyzed AI market trends showing 40% growth")
    ctx.add_task("task-2", "Competitor Analysis", "Studied LangChain's approach to context management")
    ctx.add_learning("learning-1", "Vector databases provide 10x faster semantic search", "research")
    
    # Get context
    context = ctx.get_context("AI market research", token_budget=500)
    print("Context:")
    print(context[:200] + "...")
    print()

def example_2_agent_context_manager():
    """Example 2: Agent context management."""
    print("[EXAMPLE 2] Agent Context Manager")
    print("=" * 40)
    
    # Create an agent context manager
    agent = create_agent_context_manager("research-agent")
    
    # Set agent goals
    agent.add_goal("goal-1", "Analyze AI market trends for 2025", priority=2.0)
    agent.add_goal("goal-2", "Research privacy-focused AI solutions", priority=1.5)
    
    # Record task results
    agent.add_task("task-1", "Market Analysis", "Found 40% growth in AI context management", success=True)
    agent.add_task("task-2", "Privacy Research", "Discovered 300% increase in privacy-focused AI adoption", success=True)
    
    # Record learnings
    agent.add_learning("learning-1", "Vector databases are becoming essential for agent memory", "market-research", importance=2.0)
    agent.add_learning("learning-2", "Privacy concerns drive 60% of enterprise AI decisions", "privacy-research", importance=1.8)
    
    # Get agent context
    context = agent.get_context("AI market trends and privacy", token_budget=1000)
    print("Agent Context:")
    print(context[:300] + "...")
    
    # Show agent stats
    stats = agent.get_stats()
    print(f"\nAgent Stats:")
    print(f"  Total Goals: {stats.get('total_goals', 0)}")
    print(f"  Active Goals: {stats.get('active_goals', 0)}")
    print(f"  Total Tasks: {stats.get('total_tasks', 0)}")
    print(f"  Successful Sessions: {stats.get('successful_sessions', 0)}")
    print()

def example_3_semantic_search():
    """Example 3: Semantic search capabilities."""
    print("[EXAMPLE 3] Semantic Search")
    print("=" * 40)
    
    agent = create_agent_context_manager("search-agent")
    
    # Add diverse content
    agent.add_task("task-1", "Database Performance", "Vector databases show 10x improvement over traditional search")
    agent.add_task("task-2", "AI Market Analysis", "Context management systems growing 40% annually")
    agent.add_task("task-3", "Privacy Research", "Privacy-focused AI solutions gaining 300% adoption")
    agent.add_learning("learning-1", "Semantic similarity search outperforms keyword matching", "performance-testing")
    
    # Search for similar content
    query = "database performance improvements"
    results = agent.search_similar(query, limit=3)
    
    print(f"Search results for: '{query}'")
    for i, result in enumerate(results, 1):
        similarity = result.get('similarity_score', 0.0)
        content = result.get('content', '')[:100]
        print(f"{i}. (similarity: {similarity:.3f}) {content}...")
    print()

def example_4_batch_operations():
    """Example 4: Batch operations for efficiency."""
    print("[EXAMPLE 4] Batch Operations")
    print("=" * 40)
    
    agent = create_agent_context_manager("batch-agent")
    
    # Batch add tasks
    tasks = [
        {"id": "batch-task-1", "name": "Task 1", "summary": "Completed analysis of AI trends", "tags": ["analysis", "ai"]},
        {"id": "batch-task-2", "name": "Task 2", "summary": "Researched vector database performance", "tags": ["research", "performance"]},
        {"id": "batch-task-3", "name": "Task 3", "summary": "Studied privacy implications", "tags": ["privacy", "study"]}
    ]
    
    agent.add_multiple_tasks(tasks)
    
    # Batch add learnings
    learnings = [
        {"id": "batch-learning-1", "content": "Vector databases are 10x faster", "source": "performance-test", "importance": 2.0},
        {"id": "batch-learning-2", "content": "Privacy is a major concern for AI adoption", "source": "survey", "importance": 1.8},
        {"id": "batch-learning-3", "content": "Context management is becoming critical", "source": "market-analysis", "importance": 1.5}
    ]
    
    agent.add_multiple_learnings(learnings)
    
    # Get comprehensive context
    context = agent.get_context("AI trends and performance", token_budget=800)
    print("Batch Context:")
    print(context[:250] + "...")
    print()

def example_5_quick_setup():
    """Example 5: Quick setup for immediate use."""
    print("[EXAMPLE 5] Quick Setup")
    print("=" * 40)
    
    # One-liner setup
    agent = quick_setup("demo-agent")
    
    # Immediate use
    agent.add_task("demo-1", "Demo Task", "This is a quick demo of the context manager")
    agent.add_learning("demo-learning", "Quick setup makes the system easy to use", "demo", importance=1.0)
    
    context = agent.get_context("demo", token_budget=300)
    print("Quick Setup Context:")
    print(context)
    print()

def main():
    """Run all examples."""
    print("[QUICK START] AI Context Manager Quick Start Examples")
    print("=" * 60)
    
    try:
        example_1_simple_task_manager()
        example_2_agent_context_manager()
        example_3_semantic_search()
        example_4_batch_operations()
        example_5_quick_setup()
        
        print("[SUCCESS] All examples completed successfully!")
        print("\n[INFO] Next Steps:")
        print("1. Try the CLI: python -m ai_context_manager.cli init")
        print("2. Run benchmarks: python benchmark_performance.py")
        print("3. Check the documentation in README.md")
        
    except Exception as e:
        print(f"[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
