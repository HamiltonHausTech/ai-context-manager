#!/usr/bin/env python3
"""
Research Assistant Demo Application

A sophisticated AI research assistant that demonstrates the power of the AI Context Manager.
This agent can research topics, remember findings, and build knowledge over time.

Features:
- Persistent memory across sessions
- Semantic search through research history
- Goal tracking and progress monitoring
- Learning from research patterns
- Multi-topic research capabilities
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Add the parent directory to the path to import ai_context_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai_context_manager.simple_api import create_agent_context_manager
from ai_context_manager.auto_config import auto_detect_and_setup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchAssistant:
    """
    AI Research Assistant using the AI Context Manager for persistent memory.
    
    This agent demonstrates:
    - Long-term memory persistence
    - Semantic search capabilities
    - Goal tracking and progress monitoring
    - Learning from research patterns
    - Multi-topic research coordination
    """
    
    def __init__(self, agent_id: str = "research-assistant", config_path: str = None):
        """Initialize the research assistant."""
        self.agent_id = agent_id
        self.agent = create_agent_context_manager(agent_id, config_path=config_path)
        self.session_start = datetime.now()
        
        # Initialize research goals
        self._setup_initial_goals()
        
        logger.info(f"Research Assistant '{agent_id}' initialized")
    
    def _setup_initial_goals(self):
        """Set up initial research goals for the assistant."""
        # Add some default research goals
        goals = [
            {
                "id": "ai-trends-2025",
                "description": "Research AI trends and developments for 2025",
                "priority": 2.0
            },
            {
                "id": "vector-databases",
                "description": "Investigate vector database technologies and use cases",
                "priority": 1.5
            },
            {
                "id": "agent-architecture",
                "description": "Study modern AI agent architectures and patterns",
                "priority": 1.8
            }
        ]
        
        for goal in goals:
            try:
                self.agent.add_goal(
                    goal["id"],
                    goal["description"],
                    priority=goal["priority"]
                )
                logger.info(f"Added research goal: {goal['description']}")
            except Exception as e:
                logger.warning(f"Failed to add goal {goal['id']}: {e}")
    
    async def research_topic(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """
        Research a specific topic and store findings.
        
        Args:
            topic: Research topic
            depth: Research depth (shallow, medium, deep)
            
        Returns:
            Research results with findings and insights
        """
        logger.info(f"Starting research on: {topic}")
        
        # Check for existing research on this topic
        existing_research = self.agent.search_similar(f"research {topic}", limit=5)
        
        # Build research plan based on depth
        research_plan = self._build_research_plan(topic, depth, existing_research)
        
        findings = []
        insights = []
        
        for step in research_plan:
            logger.info(f"Research step: {step['action']}")
            
            # Simulate research step (in real app, this would call APIs, scrape web, etc.)
            result = await self._simulate_research_step(step, topic)
            
            # Record the research step
            task_id = f"research-{topic.lower().replace(' ', '-')}-{step['id']}"
            self.agent.add_task(
                task_id=task_id,
                task_name=step["action"],
                result=f"Research finding: {result['finding']}",
                success=result["success"],
                tags=["research", topic.lower().replace(" ", "-"), step["type"]]
            )
            
            findings.append(result["finding"])
            
            # Extract insights if this is a significant finding
            if result["significance"] > 0.7:
                insight = self._extract_insight(result, topic)
                if insight:
                    insight_id = f"insight-{topic.lower().replace(' ', '-')}-{step['id']}"
                    self.agent.add_learning(
                        learning_id=insight_id,
                        content=insight,
                        source=f"research-{topic}",
                        importance=result["significance"],
                        tags=["insight", topic.lower().replace(" ", "-")]
                    )
                    insights.append(insight)
        
        # Update research goal progress
        self._update_research_progress(topic, findings)
        
        return {
            "topic": topic,
            "findings": findings,
            "insights": insights,
            "existing_research_count": len(existing_research),
            "new_findings_count": len(findings),
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_research_plan(self, topic: str, depth: str, existing_research: List[Dict]) -> List[Dict]:
        """Build a research plan based on topic, depth, and existing research."""
        plans = {
            "shallow": [
                {"id": "overview", "action": f"Get overview of {topic}", "type": "background"},
                {"id": "current", "action": f"Find current developments in {topic}", "type": "current"},
                {"id": "summary", "action": f"Summarize key points about {topic}", "type": "synthesis"}
            ],
            "medium": [
                {"id": "overview", "action": f"Get comprehensive overview of {topic}", "type": "background"},
                {"id": "history", "action": f"Research historical context of {topic}", "type": "historical"},
                {"id": "current", "action": f"Find current developments and trends in {topic}", "type": "current"},
                {"id": "players", "action": f"Identify key players and organizations in {topic}", "type": "stakeholders"},
                {"id": "challenges", "action": f"Identify challenges and opportunities in {topic}", "type": "analysis"},
                {"id": "summary", "action": f"Synthesize comprehensive summary of {topic}", "type": "synthesis"}
            ],
            "deep": [
                {"id": "overview", "action": f"Get comprehensive overview of {topic}", "type": "background"},
                {"id": "history", "action": f"Research detailed historical context of {topic}", "type": "historical"},
                {"id": "current", "action": f"Find current developments and trends in {topic}", "type": "current"},
                {"id": "players", "action": f"Identify key players and organizations in {topic}", "type": "stakeholders"},
                {"id": "challenges", "action": f"Identify challenges and opportunities in {topic}", "type": "analysis"},
                {"id": "future", "action": f"Research future outlook and predictions for {topic}", "type": "future"},
                {"id": "comparison", "action": f"Compare {topic} with related fields", "type": "comparative"},
                {"id": "case_studies", "action": f"Find case studies and examples of {topic}", "type": "examples"},
                {"id": "summary", "action": f"Create comprehensive synthesis of {topic}", "type": "synthesis"}
            ]
        }
        
        base_plan = plans.get(depth, plans["medium"])
        
        # Add steps to address gaps in existing research
        if len(existing_research) < 3:
            base_plan.insert(0, {
                "id": "gap_analysis",
                "action": f"Identify research gaps in {topic}",
                "type": "gap_analysis"
            })
        
        return base_plan
    
    async def _simulate_research_step(self, step: Dict, topic: str) -> Dict[str, Any]:
        """Simulate a research step (in real app, this would do actual research)."""
        # Simulate research delay
        await asyncio.sleep(0.5)
        
        # Generate realistic research findings based on step type
        findings_by_type = {
            "background": f"{topic} is a rapidly evolving field with significant impact on technology and society.",
            "historical": f"The history of {topic} dates back several decades, with major breakthroughs in recent years.",
            "current": f"Current developments in {topic} show strong growth and adoption across industries.",
            "stakeholders": f"Key players in {topic} include major tech companies, startups, and research institutions.",
            "analysis": f"Major challenges in {topic} include scalability, cost, and complexity, but opportunities are abundant.",
            "future": f"The future of {topic} looks promising with emerging technologies and growing demand.",
            "comparative": f"{topic} compares favorably to related technologies in terms of performance and adoption.",
            "examples": f"Notable examples of {topic} in practice include successful implementations in various industries.",
            "synthesis": f"Overall, {topic} represents a significant opportunity with both challenges and potential.",
            "gap_analysis": f"Research gaps in {topic} include limited real-world case studies and long-term impact analysis."
        }
        
        finding = findings_by_type.get(step["type"], f"Research finding about {topic}: {step['action']}")
        
        # Add some variation and realism
        import random
        variations = [
            f"Recent studies show that {finding.lower()}",
            f"Industry experts believe that {finding.lower()}",
            f"Market analysis indicates that {finding.lower()}",
            f"Research suggests that {finding.lower()}",
            f"Data shows that {finding.lower()}"
        ]
        
        finding = random.choice(variations)
        
        return {
            "finding": finding,
            "success": random.random() > 0.1,  # 90% success rate
            "significance": random.uniform(0.3, 1.0),  # Random significance
            "confidence": random.uniform(0.6, 0.95)  # Random confidence
        }
    
    def _extract_insight(self, result: Dict, topic: str) -> Optional[str]:
        """Extract insights from significant research findings."""
        if result["significance"] > 0.8:
            return f"Key insight about {topic}: {result['finding']} (Confidence: {result['confidence']:.2f})"
        elif result["significance"] > 0.7:
            return f"Notable finding about {topic}: {result['finding']}"
        return None
    
    def _update_research_progress(self, topic: str, findings: List[str]):
        """Update progress on relevant research goals."""
        # Find goals related to this topic
        stats = self.agent.get_stats()
        if "total_goals" in stats:
            # Update goal progress based on findings
            progress = min(len(findings) * 0.2, 1.0)  # 20% per finding, max 100%
            
            # Try to find and update relevant goals
            goal_ids = ["ai-trends-2025", "vector-databases", "agent-architecture"]
            for goal_id in goal_ids:
                if topic.lower() in goal_id.lower() or any(keyword in topic.lower() for keyword in goal_id.lower().split("-")):
                    try:
                        self.agent.update_goal_progress(goal_id, progress)
                        logger.info(f"Updated goal {goal_id} progress to {progress:.1%}")
                    except Exception as e:
                        logger.warning(f"Failed to update goal {goal_id}: {e}")
    
    def get_research_summary(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of research conducted."""
        if topic:
            # Get topic-specific research
            context = self.agent.get_context(f"research {topic}", token_budget=1000)
            similar_research = self.agent.search_similar(f"research {topic}", limit=10)
        else:
            # Get overall research summary
            context = self.agent.get_context("research findings", token_budget=1500)
            similar_research = self.agent.search_similar("research", limit=15)
        
        stats = self.agent.get_stats()
        
        return {
            "topic": topic or "All Research",
            "context_summary": context[:500] + "..." if len(context) > 500 else context,
            "research_count": len(similar_research),
            "agent_stats": stats,
            "session_duration": (datetime.now() - self.session_start).total_seconds() / 60,
            "timestamp": datetime.now().isoformat()
        }
    
    def ask_question(self, question: str) -> str:
        """Answer a question using accumulated research knowledge."""
        logger.info(f"Answering question: {question}")
        
        # Search for relevant research
        relevant_research = self.agent.search_similar(question, limit=5)
        
        if not relevant_research:
            return f"I don't have enough research on this topic yet. Would you like me to research '{question}'?"
        
        # Build answer from relevant research
        answer_parts = [f"Based on my research, here's what I can tell you about your question:\n"]
        
        for i, research in enumerate(relevant_research, 1):
            similarity = research.get("similarity_score", 0.0)
            content = research.get("content", "")
            answer_parts.append(f"{i}. (Relevance: {similarity:.2f}) {content}")
        
        return "\n".join(answer_parts)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status and statistics."""
        stats = self.agent.get_stats()
        
        return {
            "agent_id": self.agent_id,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "total_goals": stats.get("total_goals", 0),
            "active_goals": stats.get("active_goals", 0),
            "completed_goals": stats.get("completed_goals", 0),
            "total_tasks": stats.get("total_tasks", 0),
            "successful_sessions": stats.get("successful_sessions", 0),
            "semantic_search_enabled": stats.get("semantic_search_enabled", False),
            "memory_store_type": stats.get("memory_store_type", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main demonstration function."""
    print("=" * 60)
    print("🤖 AI Research Assistant Demo")
    print("=" * 60)
    print()
    
    # Initialize the research assistant
    assistant = ResearchAssistant("demo-research-assistant")
    
    print("📊 Initial Agent Status:")
    status = assistant.get_agent_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # Demonstrate research capabilities
    print("🔍 Research Demonstration:")
    print("-" * 30)
    
    # Research multiple topics
    topics = [
        ("AI Agent Memory Systems", "medium"),
        ("Vector Database Performance", "shallow"),
        ("Context Management Patterns", "deep")
    ]
    
    for topic, depth in topics:
        print(f"\n📚 Researching: {topic} (depth: {depth})")
        result = await assistant.research_topic(topic, depth)
        
        print(f"  ✅ Found {result['new_findings_count']} new findings")
        print(f"  💡 Generated {len(result['insights'])} insights")
        print(f"  📖 Found {result['existing_research_count']} existing research items")
        
        # Show a sample finding
        if result['findings']:
            print(f"  📝 Sample finding: {result['findings'][0][:100]}...")
    
    print("\n" + "=" * 60)
    print("🧠 Knowledge Demonstration:")
    print("=" * 60)
    
    # Ask questions to demonstrate knowledge
    questions = [
        "What are the key trends in AI agent development?",
        "How do vector databases improve performance?",
        "What are the main challenges in context management?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = assistant.ask_question(question)
        print(f"💬 Answer: {answer[:200]}...")
    
    print("\n" + "=" * 60)
    print("📊 Final Agent Status:")
    print("=" * 60)
    
    final_status = assistant.get_agent_status()
    for key, value in final_status.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("\nThe Research Assistant has demonstrated:")
    print("  ✅ Persistent memory across research sessions")
    print("  ✅ Semantic search through accumulated knowledge")
    print("  ✅ Goal tracking and progress monitoring")
    print("  ✅ Learning from research patterns")
    print("  ✅ Multi-topic research coordination")
    print("\nTry running the assistant again to see persistent memory in action!")


if __name__ == "__main__":
    asyncio.run(main())
