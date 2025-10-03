"""
Simplified API for easy usage of the AI Context Manager
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .context_manager import ContextManager
from .config import Config
from .utils import load_stores_from_config, load_summarizer
from .feedback import Feedback
from .components import ContextComponent, TaskSummaryComponent

# Optional imports with fallbacks
try:
    from .semantic_context_manager import SemanticContextManager
except ImportError:
    SemanticContextManager = None

try:
    from .agent_context_manager import AgentContextManager
except ImportError:
    AgentContextManager = None

try:
    from .semantic_agent_context_manager import SemanticAgentContextManager
except ImportError:
    SemanticAgentContextManager = None

try:
    from .components import LongTermMemoryComponent
except ImportError:
    LongTermMemoryComponent = None

logger = logging.getLogger(__name__)

class ContextManagerBuilder:
    """Builder pattern for easy context manager creation."""
    
    def __init__(self):
        self.config_path = "config.toml"
        self.agent_id = None
        self.use_semantic = True
        self.custom_config = None
        
    def with_config(self, config_path: str = "config.toml"):
        """Set configuration file path."""
        self.config_path = config_path
        return self
    
    def with_custom_config(self, config_dict: Dict[str, Any]):
        """Use custom configuration dictionary."""
        self.custom_config = config_dict
        return self
    
    def for_agent(self, agent_id: str):
        """Create context manager for a specific agent."""
        self.agent_id = agent_id
        return self
    
    def without_semantic_search(self):
        """Disable semantic search (use traditional retrieval)."""
        self.use_semantic = False
        return self
    
    def build(self) -> Union[ContextManager, AgentContextManager]:
        """Build the context manager."""
        try:
            # Load configuration
            if self.custom_config:
                config_data = self.custom_config
                config_obj = None
            else:
                config_obj = Config(self.config_path)
                config_data = config_obj.data
            
            # Load stores and summarizer
            feedback_store, memory_store = load_stores_from_config(config_data)
            feedback = Feedback(store=feedback_store)
            summarizer = load_summarizer(config_obj if config_obj else config_data)
            
            # Create base context manager
            if self.use_semantic and SemanticContextManager is not None:
                base_ctx = SemanticContextManager(
                    feedback=feedback,
                    memory_store=memory_store,
                    summarizer=summarizer,
                    config=config_data
                )
            else:
                base_ctx = ContextManager(
                    feedback=feedback,
                    memory_store=memory_store,
                    summarizer=summarizer,
                    config=config_data
                )
            
            # Return appropriate manager
            if self.agent_id:
                if self.use_semantic and SemanticAgentContextManager is not None:
                    return SemanticAgentContextManager(self.agent_id, base_ctx)
                elif AgentContextManager is not None:
                    return AgentContextManager(self.agent_id, base_ctx)
                else:
                    # Fallback to base context manager if agent managers not available
                    logger.warning("Agent context managers not available, using base context manager")
                    return base_ctx
            else:
                return base_ctx
                
        except Exception as e:
            logger.error(f"Failed to build context manager: {e}")
            raise


class SimpleContextManager:
    """Simplified interface for common operations."""
    
    def __init__(self, context_manager):
        self.ctx = context_manager
        self.is_agent = hasattr(context_manager, 'agent_id')
    
    # === Simple Component Creation ===
    
    def add_task(self, task_id: str, task_name: str, summary: str, 
                tags: Optional[List[str]] = None, success: bool = True):
        """Add a task result."""
        if self.is_agent:
            self.ctx.record_task_result(task_id, task_name, summary, success, tags)
        else:
            task = TaskSummaryComponent(
                id=task_id,
                task_name=task_name,
                summary=summary,
                score=2.0 if success else 0.5,
                tags=tags or []
            )
            self.ctx.register_component(task)
    
    def add_learning(self, learning_id: str, content: str, source: str,
                    importance: float = 1.0, tags: Optional[List[str]] = None):
        """Add learned information."""
        if self.is_agent:
            self.ctx.record_learning(learning_id, content, source, importance, tags)
        else:
            learning = LongTermMemoryComponent(
                id=learning_id,
                content=content,
                source=source,
                timestamp=datetime.utcnow().isoformat(),
                score=importance,
                tags=tags or []
            )
            self.ctx.register_component(learning)
    
    def add_goal(self, goal_id: str, goal_description: str, 
                priority: float = 1.0, deadline: Optional[str] = None,
                tags: Optional[List[str]] = None):
        """Add an agent goal (only for agent context managers)."""
        if not self.is_agent:
            raise ValueError("Goals can only be added to agent context managers")
        return self.ctx.add_goal(goal_id, goal_description, priority, deadline, tags)
    
    # === Simple Context Retrieval ===
    
    def get_context(self, query: Optional[str] = None, 
                   tags: Optional[List[str]] = None,
                   token_budget: int = 2000) -> str:
        """Get context using the most appropriate method."""
        if query and hasattr(self.ctx, 'get_semantic_context'):
            # Use semantic search if available and query provided
            return self.ctx.get_semantic_context(query, token_budget)
        elif self.is_agent:
            # Use agent-specific context retrieval
            return self.ctx.get_agent_context(token_budget=token_budget)
        else:
            # Use traditional tag-based retrieval
            return self.ctx.get_context(
                include_tags=tags,
                token_budget=token_budget,
                summarize_if_needed=True
            )
    
    def search_similar(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for similar content."""
        if hasattr(self.ctx, 'search_similar_components'):
            return self.ctx.search_similar_components(query, n_results=limit)
        else:
            # Fallback to basic search
            if hasattr(self.ctx, 'base_ctx'):
                components = list(self.ctx.base_ctx.components.values())
            else:
                components = list(self.ctx.ctx.components.values())
            # Simple text matching (could be improved)
            similar = []
            query_lower = query.lower()
            for comp in components:
                content = comp.get_content().lower()
                if any(word in content for word in query_lower.split()):
                    similar.append({
                        "id": comp.id,
                        "type": comp.__class__.__name__,
                        "content": comp.get_content(),
                        "tags": comp.tags,
                        "similarity_score": 0.5  # Placeholder
                    })
            return similar[:limit]
    
    # === Simple Statistics ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics."""
        if self.is_agent and hasattr(self.ctx, 'get_agent_stats'):
            return self.ctx.get_agent_stats()
        else:
            if hasattr(self.ctx, 'base_ctx'):
                components = self.ctx.base_ctx.components
            elif hasattr(self.ctx, 'ctx'):
                components = self.ctx.ctx.components
            else:
                components = self.ctx.components
            return {
                "total_components": len(components),
                "component_types": {
                    comp.__class__.__name__: sum(1 for c in components.values() 
                                               if c.__class__.__name__ == comp.__class__.__name__)
                    for comp in components.values()
                }
            }
    
    # === Batch Operations ===
    
    def add_multiple_tasks(self, tasks: List[Dict[str, Any]]):
        """Add multiple tasks at once."""
        for task in tasks:
            self.add_task(
                task["id"], 
                task["name"], 
                task["summary"], 
                task.get("tags"), 
                task.get("success", True)
            )
    
    def add_multiple_learnings(self, learnings: List[Dict[str, Any]]):
        """Add multiple learnings at once."""
        for learning in learnings:
            self.add_learning(
                learning["id"],
                learning["content"],
                learning["source"],
                learning.get("importance", 1.0),
                learning.get("tags")
            )


# === Convenience Functions ===

def create_context_manager(config_path: str = "config.toml", 
                          agent_id: Optional[str] = None,
                          use_semantic: bool = True) -> SimpleContextManager:
    """Create a context manager with sensible defaults."""
    builder = ContextManagerBuilder().with_config(config_path)
    
    if agent_id:
        builder = builder.for_agent(agent_id)
    
    if not use_semantic:
        builder = builder.without_semantic_search()
    
    ctx = builder.build()
    return SimpleContextManager(ctx)


def create_agent_context_manager(agent_id: str, 
                                config_path: str = "config.toml",
                                use_semantic: bool = True) -> SimpleContextManager:
    """Create an agent context manager with sensible defaults."""
    return create_context_manager(config_path, agent_id, use_semantic)


# === Quick Start Functions ===

def quick_setup(agent_id: str = "default-agent") -> SimpleContextManager:
    """Quick setup for immediate use."""
    return create_agent_context_manager(agent_id)


def quick_task_manager() -> SimpleContextManager:
    """Quick setup for simple task management."""
    return create_context_manager()
