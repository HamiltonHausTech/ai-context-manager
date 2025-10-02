"""
Semantic Agent Context Manager - Enhanced agent context management with vector database
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .semantic_context_manager import SemanticContextManager
from .components.agent_goal import AgentGoalComponent, AgentSessionComponent
from .components.task_summary import TaskSummaryComponent
from .components.longterm_summary import LongTermMemoryComponent

logger = logging.getLogger(__name__)

class SemanticAgentContextManager:
    """
    Enhanced agent context manager with semantic similarity search capabilities.
    Provides intelligent context retrieval for long-running AI agents.
    """
    
    def __init__(self, agent_id: str, base_context_manager: Optional[SemanticContextManager] = None):
        self.agent_id = agent_id
        self.ctx = base_context_manager or SemanticContextManager()
        self.start_time = datetime.utcnow()
        
        # Register agent-specific components
        self._register_agent_components()
    
    def _register_agent_components(self):
        """Register agent-specific components if not already present."""
        # Add agent session for this startup
        session = AgentSessionComponent(
            id=f"session-{self.agent_id}-{int(self.start_time.timestamp())}",
            agent_id=self.agent_id,
            session_type="startup",
            summary=f"Agent {self.agent_id} started at {self.start_time.isoformat()}",
            duration_minutes=0.0,
            success=True,
            tags=["agent", "session", "startup"]
        )
        self.ctx.register_component(session)
    
    def add_goal(self, goal_id: str, goal_description: str, 
                 priority: float = 1.0, deadline: Optional[str] = None,
                 tags: Optional[List[str]] = None) -> AgentGoalComponent:
        """Add a new agent goal."""
        goal = AgentGoalComponent(
            id=goal_id,
            goal_description=goal_description,
            agent_id=self.agent_id,
            priority=priority,
            deadline=deadline,
            tags=(tags or []) + ["agent", "goal"]
        )
        self.ctx.register_component(goal)
        logger.info(f"Added goal: {goal_description}")
        return goal
    
    def update_goal_progress(self, goal_id: str, progress: float, 
                           status: Optional[str] = None):
        """Update the progress of an agent goal."""
        goal = self.ctx.components.get(goal_id)
        if goal and isinstance(goal, AgentGoalComponent):
            goal.update_progress(progress, status)
            self.ctx.save_component_to_memory(goal)
            logger.info(f"Updated goal {goal_id}: {progress:.1%} progress")
        else:
            logger.warning(f"Goal {goal_id} not found")
    
    def record_task_result(self, task_id: str, task_name: str, result: str,
                          success: bool = True, tags: Optional[List[str]] = None):
        """Record the result of a task execution."""
        task = TaskSummaryComponent(
            id=task_id,
            task_name=task_name,
            summary=result,
            score=2.0 if success else 0.5,
            tags=(tags or []) + ["agent", "task", "result"]
        )
        self.ctx.register_component(task)
        
        # Record session for this task
        session = AgentSessionComponent(
            id=f"session-{task_id}",
            agent_id=self.agent_id,
            session_type="task-execution",
            summary=f"Executed task: {task_name}",
            duration_minutes=0.0,
            success=success,
            tags=["agent", "session", "task-execution"]
        )
        self.ctx.register_component(session)
        
        logger.info(f"Recorded task result: {task_name} ({'SUCCESS' if success else 'FAILED'})")
    
    def record_learning(self, learning_id: str, content: str, source: str,
                       importance: float = 1.0, tags: Optional[List[str]] = None):
        """Record learned information or insights."""
        learning = LongTermMemoryComponent(
            id=learning_id,
            content=content,
            source=source,
            timestamp=datetime.utcnow().isoformat(),
            score=importance,
            tags=(tags or []) + ["agent", "learning", "insight"]
        )
        self.ctx.register_component(learning)
        logger.info(f"Recorded learning: {content[:50]}...")
    
    def get_semantic_context(self, query: str, token_budget: int = 2000,
                           max_components: int = 20) -> str:
        """
        Get context using semantic similarity search.
        Automatically filters for agent-specific content.
        """
        agent_tags = [self.agent_id, "agent"]
        return self.ctx.get_semantic_context(
            query=query,
            token_budget=token_budget,
            max_components=max_components,
            include_tags=agent_tags
        )
    
    def get_agent_context(self, task_type: Optional[str] = None,
                         include_goals: bool = True,
                         include_recent_tasks: bool = True,
                         include_learnings: bool = True,
                         token_budget: int = 2000,
                         use_semantic_search: bool = True) -> str:
        """
        Get comprehensive context for the agent's current situation.
        Can use semantic search or traditional tag-based retrieval.
        """
        if use_semantic_search and hasattr(self.ctx, 'semantic_retriever') and self.ctx.semantic_retriever:
            return self._get_semantic_agent_context(task_type, token_budget)
        else:
            return self._get_traditional_agent_context(task_type, include_goals, include_recent_tasks, include_learnings, token_budget)
    
    def _get_semantic_agent_context(self, task_type: Optional[str], token_budget: int) -> str:
        """Get agent context using semantic search."""
        # Build semantic query
        query_parts = [f"agent {self.agent_id}"]
        if task_type:
            query_parts.append(task_type)
        
        # Add goal-related terms
        active_goals = self.get_active_goals()
        if active_goals:
            query_parts.append("goals objectives")
        
        semantic_query = " ".join(query_parts)
        
        return self.get_semantic_context(semantic_query, token_budget)
    
    def _get_traditional_agent_context(self, task_type: Optional[str], include_goals: bool,
                                     include_recent_tasks: bool, include_learnings: bool,
                                     token_budget: int) -> str:
        """Get agent context using traditional tag-based retrieval."""
        tags = ["agent"]
        if task_type:
            tags.append(task_type)
        
        # Build context parts
        context_parts = []
        
        if include_goals:
            goal_context = self.ctx.get_context(
                include_tags=tags + ["goal"],
                token_budget=token_budget // 3 if include_recent_tasks or include_learnings else token_budget // 2,
                summarize_if_needed=True
            )
            if goal_context:
                context_parts.append(f"=== AGENT GOALS ===\n{goal_context}")
        
        if include_recent_tasks:
            task_context = self.ctx.get_context(
                include_tags=tags + ["task", "result"],
                token_budget=token_budget // 3 if include_goals or include_learnings else token_budget // 2,
                summarize_if_needed=True
            )
            if task_context:
                context_parts.append(f"=== RECENT TASKS ===\n{task_context}")
        
        if include_learnings:
            learning_context = self.ctx.get_context(
                include_tags=tags + ["learning", "insight"],
                token_budget=token_budget // 3 if include_goals or include_recent_tasks else token_budget // 2,
                summarize_if_needed=True
            )
            if learning_context:
                context_parts.append(f"=== LEARNED INSIGHTS ===\n{learning_context}")
        
        return "\n\n".join(context_parts)
    
    def search_similar_learnings(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for similar learned insights."""
        return self.ctx.search_similar_components(
            query=query,
            n_results=n_results,
            include_types=["LongTermMemoryComponent"],
            include_tags=[self.agent_id, "learning"]
        )
    
    def find_similar_goals(self, goal_description: str, n_results: int = 5) -> List[Dict]:
        """Find similar goals to avoid duplication."""
        return self.ctx.search_similar_components(
            query=goal_description,
            n_results=n_results,
            include_types=["AgentGoalComponent"],
            include_tags=[self.agent_id, "goal"]
        )
    
    def get_active_goals(self) -> List[AgentGoalComponent]:
        """Get all active goals for this agent."""
        goals = []
        for component in self.ctx.components.values():
            if (isinstance(component, AgentGoalComponent) and 
                component.agent_id == self.agent_id and 
                component.status == "active"):
                goals.append(component)
        return goals
    
    def get_overdue_goals(self) -> List[AgentGoalComponent]:
        """Get all overdue goals for this agent."""
        overdue = []
        for component in self.ctx.components.values():
            if (isinstance(component, AgentGoalComponent) and 
                component.agent_id == self.agent_id and 
                component.is_overdue()):
                overdue.append(component)
        return overdue
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's operation."""
        goals = []
        sessions = []
        tasks = []
        
        for component in self.ctx.components.values():
            if isinstance(component, AgentGoalComponent) and component.agent_id == self.agent_id:
                goals.append(component)
            elif isinstance(component, AgentSessionComponent) and component.agent_id == self.agent_id:
                sessions.append(component)
            elif hasattr(component, 'tags') and 'task' in component.tags and 'agent' in component.tags:
                tasks.append(component)
        
        active_goals = len([g for g in goals if g.status == "active"])
        completed_goals = len([g for g in goals if g.status == "completed"])
        successful_sessions = len([s for s in sessions if s.success])
        
        stats = {
            "agent_id": self.agent_id,
            "uptime_minutes": (datetime.utcnow() - self.start_time).total_seconds() / 60,
            "total_goals": len(goals),
            "active_goals": active_goals,
            "completed_goals": completed_goals,
            "total_sessions": len(sessions),
            "successful_sessions": successful_sessions,
            "total_tasks": len(tasks),
            "overdue_goals": len(self.get_overdue_goals()),
            "semantic_search_enabled": hasattr(self.ctx, 'semantic_retriever') and self.ctx.semantic_retriever is not None
        }
        
        # Add memory store stats
        if hasattr(self.ctx, 'get_memory_stats'):
            memory_stats = self.ctx.get_memory_stats()
            stats.update(memory_stats)
        
        return stats
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Remove old session data to prevent memory bloat."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        removed_count = 0
        
        to_remove = []
        for component_id, component in self.ctx.components.items():
            if (isinstance(component, AgentSessionComponent) and 
                component.agent_id == self.agent_id):
                try:
                    session_date = datetime.fromisoformat(component.timestamp)
                    if session_date < cutoff_date:
                        to_remove.append(component_id)
                except ValueError:
                    # Invalid timestamp, remove it
                    to_remove.append(component_id)
        
        for component_id in to_remove:
            self.ctx.remove_component(component_id)
            removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old sessions")
        return removed_count
