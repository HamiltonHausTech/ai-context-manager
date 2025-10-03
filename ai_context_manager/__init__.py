"""
AI Context Manager - Enterprise-grade context management for AI agents.

This package provides a comprehensive context management system designed for
long-running AI agents that need to maintain and consolidate context over time
without losing important information about their goals.

Key Features:
- Token-aware context budgeting
- Semantic search with vector databases
- Persistent memory storage
- Feedback-driven learning
- Agent goal tracking
- Multiple summarization backends
- Production-ready PostgreSQL + pgvector support

Quick Start:
    from ai_context_manager import ContextManager
    from ai_context_manager.simple_api import create_agent_context_manager
    
    # Create a context manager
    ctx = ContextManager()
    
    # Or use the simplified API
    agent = create_agent_context_manager("my-agent")
"""

# Core imports - these should always work
from .context_manager import ContextManager
from .config import Config
from .feedback import Feedback

# Version information
__version__ = "0.2.0"
__author__ = "AI Context Manager Team"
__email__ = "team@ai-context-manager.com"

# Main exports that should always be available
__all__ = [
    "ContextManager",
    "Config",
    "Feedback",
    "__version__",
    "__author__",
    "__email__",
]

# Optional imports with graceful fallbacks
try:
    from .simple_api import create_context_manager, create_agent_context_manager
    __all__.extend(["create_context_manager", "create_agent_context_manager"])
except ImportError as e:
    print(f"Warning: Could not import simple_api functions: {e}")

try:
    from .agent_context_manager import AgentContextManager
    __all__.append("AgentContextManager")
except ImportError:
    pass

try:
    from .semantic_context_manager import SemanticContextManager
    __all__.append("SemanticContextManager")
except ImportError:
    pass

try:
    from .semantic_agent_context_manager import SemanticAgentContextManager
    __all__.append("SemanticAgentContextManager")
except ImportError:
    pass

try:
    from .store.vector_memory import VectorMemoryStore
    __all__.append("VectorMemoryStore")
except ImportError:
    pass

try:
    from .store.postgres_vector_memory import PostgreSQLVectorMemoryStore
    __all__.append("PostgreSQLVectorMemoryStore")
except ImportError:
    pass
