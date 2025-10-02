"""
Agent Goal Component - Tracks long-term agent objectives and progress
"""

from ai_context_manager.components import ContextComponent
from typing import Dict, Any, List, Optional
from datetime import datetime

class AgentGoalComponent(ContextComponent):
    """Represents a long-term goal or objective for an AI agent."""
    
    def __init__(self, id: str, goal_description: str, agent_id: str, 
                 priority: float = 1.0, status: str = "active",
                 progress: float = 0.0, deadline: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        super().__init__(id, tags or ["agent", "goal"])
        self.goal_description = goal_description
        self.agent_id = agent_id
        self.priority = priority
        self.status = status  # active, completed, paused, failed
        self.progress = progress  # 0.0 to 1.0
        self.deadline = deadline
        self.created_at = datetime.utcnow().isoformat()
        self.last_updated = datetime.utcnow().isoformat()

    def load_content(self) -> str:
        """Generate content describing the agent goal."""
        deadline_str = f" (deadline: {self.deadline})" if self.deadline else ""
        return (
            f"Agent Goal: {self.goal_description}\n"
            f"Agent ID: {self.agent_id}\n"
            f"Status: {self.status}\n"
            f"Progress: {self.progress:.1%}\n"
            f"Priority: {self.priority:.1f}{deadline_str}\n"
            f"Created: {self.created_at}\n"
            f"Updated: {self.last_updated}"
        )

    def score(self) -> float:
        """Calculate goal importance score based on priority, status, and recency."""
        base_score = self.priority
        
        # Boost active goals
        if self.status == "active":
            base_score *= 1.5
        elif self.status == "completed":
            base_score *= 0.3  # Completed goals are less important
        elif self.status == "failed":
            base_score *= 0.1
        
        # Boost goals with progress (agent is actively working on them)
        if self.progress > 0:
            base_score *= (1 + self.progress)
        
        return base_score

    def update_progress(self, progress: float, status: Optional[str] = None):
        """Update goal progress and status."""
        self.progress = max(0.0, min(1.0, progress))
        if status:
            self.status = status
        self.last_updated = datetime.utcnow().isoformat()

    def is_overdue(self) -> bool:
        """Check if goal is past its deadline."""
        if not self.deadline:
            return False
        try:
            deadline_dt = datetime.fromisoformat(self.deadline)
            return datetime.utcnow() > deadline_dt
        except ValueError:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get extended metadata for agent goals."""
        metadata = super().get_metadata()
        metadata.update({
            "agent_id": self.agent_id,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "deadline": self.deadline,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "is_overdue": self.is_overdue()
        })
        return metadata


class AgentSessionComponent(ContextComponent):
    """Represents a session or episode in an agent's operation."""
    
    def __init__(self, id: str, agent_id: str, session_type: str,
                 summary: str, duration_minutes: float,
                 success: bool = True, tags: Optional[List[str]] = None):
        super().__init__(id, tags or ["agent", "session"])
        self.agent_id = agent_id
        self.session_type = session_type
        self.summary = summary
        self.duration_minutes = duration_minutes
        self.success = success
        self.timestamp = datetime.utcnow().isoformat()

    def load_content(self) -> str:
        """Generate content describing the agent session."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"Agent Session: {self.session_type}\n"
            f"Agent ID: {self.agent_id}\n"
            f"Status: {status}\n"
            f"Duration: {self.duration_minutes:.1f} minutes\n"
            f"Summary: {self.summary}\n"
            f"Timestamp: {self.timestamp}"
        )

    def score(self) -> float:
        """Calculate session importance score."""
        base_score = 1.0
        
        # Boost successful sessions
        if self.success:
            base_score *= 1.2
        else:
            base_score *= 0.8  # Failed sessions still have learning value
        
        # Boost longer sessions (more significant work)
        if self.duration_minutes > 60:  # Sessions over 1 hour
            base_score *= 1.3
        elif self.duration_minutes > 30:  # Sessions over 30 minutes
            base_score *= 1.1
        
        return base_score
