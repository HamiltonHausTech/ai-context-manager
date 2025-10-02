import logging
import os

from typing import Dict,Any,Optional

from ai_context_manager.components import (
    ContextComponent, LongTermMemoryComponent, UserProfileComponent, 
    TaskSummaryComponent, AgentGoalComponent, AgentSessionComponent
)
from ai_context_manager.store.json_store import JSONFeedbackStore
from ai_context_manager.store.json_memory import JSONMemoryStore
from ai_context_manager.store.vector_memory import VectorMemoryStore
from ai_context_manager.config import Config
from ai_context_manager.store.sqlite_store import SQLiteFeedbackStore
from ai_context_manager.summarizers import (
    NaiveSummarizer,
    OpenAISummarizer,
    OllamaSummarizer,
    AutoFallbackSummarizer,
)

warned = False

try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    enc = None

def estimate_tokens(text: str) -> int:
    global warned
    if enc:
        return len(enc.encode(text))
    if not warned:
        logging.warning("⚠️ Falling back to naive token estimation. Install tiktoken for accuracy.")
        warned = True
    return len(text.split())

def component_from_dict(id: str, data: Dict[str, Any]) -> Optional[ContextComponent]:
    typ = data.get("type")
    tags = data.get("tags", [])
    content = data.get("content", "")

    if typ == "LongTermMemoryComponent":
        return LongTermMemoryComponent(
            id=id,
            content=content,
            source="jsonstore",
            timestamp="",
            tags=tags
        )
    elif typ == "TaskSummaryComponent":
        return TaskSummaryComponent(
            id=id,
            task_name="Recovered",
            summary=content,
            tags=tags
        )
    elif typ == "UserProfileComponent":
        return UserProfileComponent(
            id=id,
            name="Recovered",
            preferences={"recovered": content},
            tags=tags
        )
    elif typ == "AgentGoalComponent":
        return AgentGoalComponent(
            id=id,
            goal_description=content,
            agent_id=data.get("agent_id", "unknown"),
            priority=data.get("priority", 1.0),
            status=data.get("status", "active"),
            progress=data.get("progress", 0.0),
            deadline=data.get("deadline"),
            tags=tags
        )
    elif typ == "AgentSessionComponent":
        return AgentSessionComponent(
            id=id,
            agent_id=data.get("agent_id", "unknown"),
            session_type=data.get("session_type", "unknown"),
            summary=content,
            duration_minutes=data.get("duration_minutes", 0.0),
            success=data.get("success", True),
            tags=tags
        )
    else:
        return None

def load_summarizer(config):
    """Load and configure summarizer based on configuration."""
    if hasattr(config, 'get'):
        # Config object
        summarizer_config = config.load_config("summarizer", {})
    else:
        # Dict
        summarizer_config = config.get("summarizer", {})
    
    summarizer_type = summarizer_config.get("type", "naive").lower()

    try:
        if summarizer_type == "openai":
            api_key = None
            # Try to get API key from config object if it's a Config instance
            if hasattr(config, 'get_api_key'):
                api_key = config.get_api_key()
            else:
                # Fallback for dict config
                api_key = summarizer_config.get("api_key")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
            return OpenAISummarizer(
                model=summarizer_config.get("model", "gpt-4"),
                api_key=api_key
            )
        elif summarizer_type == "ollama":
            # Get Ollama host from environment variable or config
            host = os.getenv("OLLAMA_HOST")
            if not host:
                host = summarizer_config.get("host", "http://localhost:11434")
            
            return OllamaSummarizer(
                model=summarizer_config.get("model", os.getenv("OLLAMA_MODEL", "mistral")),
                host=host,
                timeout=summarizer_config.get("timeout", int(os.getenv("OLLAMA_TIMEOUT", "30")))
            )
        elif summarizer_type == "auto_fallback":
            logging.info("Using auto-fallback summarizer (tries Ollama, falls back to naive)")
            # Get Ollama host from environment variable or config
            host = os.getenv("OLLAMA_HOST")
            if not host:
                host = summarizer_config.get("host", "http://localhost:11434")
            
            return AutoFallbackSummarizer(
                model=summarizer_config.get("model", os.getenv("OLLAMA_MODEL", "mistral")),
                host=host,
                timeout=summarizer_config.get("timeout", int(os.getenv("OLLAMA_TIMEOUT", "30"))),
                health_check_timeout=summarizer_config.get("health_check_timeout", 5)
            )
        else:
            logging.info(f"Using naive summarizer (type: {summarizer_type})")
            return NaiveSummarizer()
    except Exception as e:
        logging.error(f"Failed to load summarizer: {e}")
        logging.info("Falling back to naive summarizer")
        return NaiveSummarizer()

def load_stores_from_config(config: Dict):
    fb_conf = config.get("feedback_store", {})
    mem_conf = config.get("memory_store", {})

    fb_type = fb_conf.get("type", "json")
    if fb_type == "sqlite":
        feedback_store = SQLiteFeedbackStore(fb_conf.get("db_path", "feedback.db"))
    else:
        feedback_store = JSONFeedbackStore(fb_conf.get("filepath", "feedback.json"))

    mem_type = mem_conf.get("type", "json")
    if mem_type == "vector":
        try:
            memory_store = VectorMemoryStore(
                collection_name=mem_conf.get("collection_name", "agent_memory"),
                persist_directory=mem_conf.get("persist_directory", "./chroma_db"),
                embedding_model=mem_conf.get("embedding_model", "all-MiniLM-L6-v2")
            )
            logging.info("Using vector database memory store")
        except ImportError as e:
            logging.warning(f"Vector database not available: {e}. Falling back to JSON store.")
            memory_store = JSONMemoryStore(mem_conf.get("filepath", "memory.json"))
    elif mem_type == "json":
        memory_store = JSONMemoryStore(mem_conf.get("filepath", "memory.json"))
    else:
        memory_store = None

    return feedback_store, memory_store
