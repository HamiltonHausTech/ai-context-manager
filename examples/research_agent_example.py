#!/usr/bin/env python3
"""
Example: Research Agent using AI Context Manager
Demonstrates how to use the system for a long-running research agent.
"""

import logging
from datetime import datetime, timedelta
from ai_context_manager.agent_context_manager import AgentContextManager
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.config import Config
from ai_context_manager.utils import load_stores_from_config, load_summarizer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def simulate_research_agent():
    """Simulate a research agent over multiple sessions."""
    
    # Initialize the agent context manager
    config = Config("config.toml")
    config_data = config.data
    feedback_store, memory_store = load_stores_from_config(config_data)
    
    # Create base context manager
    from ai_context_manager.feedback import Feedback
    feedback = Feedback(store=feedback_store)
    
    base_ctx = ContextManager(
        feedback=feedback,
        memory_store=memory_store,
        summarizer=load_summarizer(config),
        config=config_data
    )
    
    # Create agent context manager
    agent_id = "research-agent-001"
    agent = AgentContextManager(agent_id, base_ctx)
    
    print("=" * 60)
    print(f"Research Agent: {agent_id}")
    print("=" * 60)
    
    # Set up research goals
    print("\n[SETUP] Setting research goals...")
    agent.add_goal(
        goal_id="market-analysis-2025",
        goal_description="Analyze emerging market trends in AI technology for 2025",
        priority=2.0,
        deadline=(datetime.now() + timedelta(days=7)).isoformat(),
        tags=["market-analysis", "ai-trends", "2025"]
    )
    
    agent.add_goal(
        goal_id="competitor-research",
        goal_description="Research key competitors in the AI context management space",
        priority=1.5,
        deadline=(datetime.now() + timedelta(days=14)).isoformat(),
        tags=["competitor-analysis", "market-research"]
    )
    
    agent.add_goal(
        goal_id="user-feedback-analysis",
        goal_description="Analyze user feedback patterns for context management systems",
        priority=1.0,
        tags=["user-research", "feedback-analysis"]
    )
    
    # Simulate research tasks
    print("\n[RESEARCH] Conducting research tasks...")
    
    # Task 1: Market trend research
    agent.record_task_result(
        task_id="market-trends-research-1",
        task_name="Research AI Market Trends Q1 2025",
        result="""
        Key findings from Q1 2025 AI market research:
        1. Context management systems are seeing 40% growth year-over-year
        2. Local LLM adoption increased 300% due to privacy concerns
        3. Auto-fallback systems like ours are becoming industry standard
        4. Token budget management is critical for cost optimization
        5. Agent memory systems are the next frontier in AI applications
        """,
        success=True,
        tags=["market-research", "ai-trends", "q1-2025"]
    )
    
    # Update goal progress
    agent.update_goal_progress("market-analysis-2025", 0.4)
    
    # Record learning
    agent.record_learning(
        learning_id="privacy-trend-insight",
        content="Privacy-focused AI systems are gaining 300% more adoption than cloud-based alternatives",
        source="market-research",
        importance=2.0,
        tags=["privacy", "adoption-trends", "insight"]
    )
    
    # Task 2: Competitor analysis
    agent.record_task_result(
        task_id="competitor-analysis-1",
        task_name="Analyze LangChain Context Management",
        result="""
        LangChain context management analysis:
        - Uses basic memory systems with limited persistence
        - No automatic fallback mechanisms
        - Token management is manual and error-prone
        - No feedback learning system
        - Our system has significant advantages in persistence and learning
        """,
        success=True,
        tags=["competitor-analysis", "langchain", "feature-comparison"]
    )
    
    agent.update_goal_progress("competitor-research", 0.3)
    
    # Task 3: User feedback analysis (simulated failure)
    agent.record_task_result(
        task_id="user-feedback-analysis-1",
        task_name="Analyze User Feedback from GitHub Issues",
        result="Failed to access GitHub API due to rate limiting. Need to implement exponential backoff.",
        success=False,
        tags=["user-research", "github-api", "technical-issue"]
    )
    
    # Record learning from failure
    agent.record_learning(
        learning_id="github-api-limits",
        content="GitHub API has strict rate limits. Need exponential backoff strategy for large data collection.",
        source="failed-task-analysis",
        importance=1.5,
        tags=["api-limits", "github", "technical-learning"]
    )
    
    # Show agent context
    print("\n[CONTEXT] Current agent context:")
    context = agent.get_agent_context(
        task_type="market-research",
        token_budget=1500
    )
    print(context)
    
    # Show agent stats
    print("\n[STATS] Agent statistics:")
    stats = agent.get_agent_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show active goals
    print("\n[GOALS] Active goals:")
    active_goals = agent.get_active_goals()
    for goal in active_goals:
        print(f"  - {goal.goal_description} ({goal.progress:.1%} complete)")
    
    # Simulate more research over time
    print("\n[CONTINUED RESEARCH] Simulating continued research...")
    
    # More market research
    agent.record_task_result(
        task_id="market-trends-research-2",
        task_name="Research AI Investment Patterns",
        result="""
        AI investment pattern analysis:
        1. Context management startups raised $2.3B in 2024
        2. Local AI infrastructure investments up 250%
        3. Privacy-focused AI solutions getting premium valuations
        4. Agent memory systems are the next funding frontier
        """,
        success=True,
        tags=["market-research", "investment", "funding-trends"]
    )
    
    agent.update_goal_progress("market-analysis-2025", 0.8)
    
    # Record key insight
    agent.record_learning(
        learning_id="funding-insight",
        content="Agent memory systems are becoming a distinct funding category, separate from general AI infrastructure",
        source="investment-research",
        importance=2.5,
        tags=["funding", "agent-memory", "market-category"]
    )
    
    # Final context retrieval
    print("\n[FINAL CONTEXT] Agent context after continued research:")
    final_context = agent.get_agent_context(
        include_goals=True,
        include_recent_tasks=True,
        include_learnings=True,
        token_budget=2000
    )
    print(final_context)
    
    # Final stats
    print("\n[FINAL STATS] Agent statistics:")
    final_stats = agent.get_agent_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    simulate_research_agent()
