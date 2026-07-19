import json
import logging
import os

from typing import Callable, Dict, Any, Optional, Tuple

from ai_context_manager.components import (
    ContextComponent, LongTermMemoryComponent, UserProfileComponent, 
    TaskSummaryComponent, AgentGoalComponent, AgentSessionComponent,
    DerivedMemoryComponent,
)
from ai_context_manager.store.json_store import JSONFeedbackStore
from ai_context_manager.store.json_memory import JSONMemoryStore
from ai_context_manager.store.vector_memory import VectorMemoryStore
from ai_context_manager.store.postgres_vector_memory import PostgreSQLVectorMemoryStore
from ai_context_manager.config import Config
from ai_context_manager.store.sqlite_store import SQLiteFeedbackStore
from ai_context_manager.store.sqlite_memory import SQLiteMemoryStore
from ai_context_manager.summarizers import (
    NaiveSummarizer,
    OpenAISummarizer,
    OllamaSummarizer,
    AutoFallbackSummarizer,
)
from ai_context_manager.tokenization import estimate_tokens
from ai_context_manager.memory import MemoryLifecycle

COMPONENT_SCHEMA_VERSION = 1
_COMPONENT_CODECS: Dict[
    str,
    Tuple[
        Callable[[ContextComponent], Dict[str, Any]],
        Callable[[str, Dict[str, Any], list], ContextComponent],
    ],
] = {}


def register_component_type(
    type_name: str,
    serializer: Callable[[ContextComponent], Dict[str, Any]],
    deserializer: Callable[[str, Dict[str, Any], list], ContextComponent],
) -> None:
    """Register serialization hooks for an application-defined component type."""
    if not type_name or not callable(serializer) or not callable(deserializer):
        raise ValueError("A type name and callable serializer/deserializer are required")
    _COMPONENT_CODECS[type_name] = (serializer, deserializer)


def component_to_dict(component: ContextComponent) -> Dict[str, Any]:
    """Serialize a component without reducing it to its rendered prompt text."""
    component_data: Dict[str, Any]
    if isinstance(component, LongTermMemoryComponent):
        component_data = {
            "content": component.content,
            "source": component.source,
            "timestamp": component.timestamp,
            "score": component._score,
        }
    elif isinstance(component, TaskSummaryComponent):
        component_data = {
            "task_name": component.task_name,
            "summary": component.summary,
            "score": component._score,
        }
    elif isinstance(component, UserProfileComponent):
        component_data = {
            "name": component.name,
            "preferences": component.preferences,
            "score": component._score,
        }
    elif isinstance(component, AgentGoalComponent):
        component_data = {
            "goal_description": component.goal_description,
            "agent_id": component.agent_id,
            "priority": component.priority,
            "status": component.status,
            "progress": component.progress,
            "deadline": component.deadline,
            "created_at": component.created_at,
            "last_updated": component.last_updated,
        }
    elif isinstance(component, AgentSessionComponent):
        component_data = {
            "agent_id": component.agent_id,
            "session_type": component.session_type,
            "summary": component.summary,
            "duration_minutes": component.duration_minutes,
            "success": component.success,
            "timestamp": component.timestamp,
        }
    elif isinstance(component, DerivedMemoryComponent):
        component_data = {
            "content": component.content,
            "derivation": component.derivation,
            "score": component._score,
        }
    else:
        codec = _COMPONENT_CODECS.get(component.__class__.__name__)
        # Preserve the previous ability to save an unregistered custom component
        # as rendered content, while registered types can round-trip losslessly.
        component_data = codec[0](component) if codec else {}

    return {
        "schema_version": COMPONENT_SCHEMA_VERSION,
        "id": component.id,
        "type": component.__class__.__name__,
        "tags": list(component.tags),
        "content": component.get_content(),
        "score": component.score(),
        "memory": component.memory.to_dict(),
        # A JSON string works in JSON, Chroma metadata, and PostgreSQL JSONB.
        "component_data": json.dumps(component_data),
    }


def _deserialize_component(id: str, data: Dict[str, Any]) -> Optional[ContextComponent]:
    """Deserialize current records and retain compatibility with legacy records."""
    typ = data.get("type")
    tags = data.get("tags", [])
    content = data.get("content", "")
    raw_component_data = data.get("component_data")
    if raw_component_data is None and isinstance(data.get("metadata"), dict):
        raw_component_data = data["metadata"].get("component_data")
    if isinstance(raw_component_data, str):
        try:
            fields = json.loads(raw_component_data)
        except (TypeError, ValueError):
            fields = {}
    elif isinstance(raw_component_data, dict):
        fields = raw_component_data
    else:
        fields = {}

    if typ == "LongTermMemoryComponent":
        return LongTermMemoryComponent(
            id=id,
            content=fields.get("content", content),
            source=fields.get("source", "jsonstore"),
            timestamp=fields.get("timestamp", ""),
            score=fields.get("score", data.get("score", 0.5)),
            tags=tags
        )
    elif typ == "TaskSummaryComponent":
        return TaskSummaryComponent(
            id=id,
            task_name=fields.get("task_name", "Recovered"),
            summary=fields.get("summary", content),
            score=fields.get("score", data.get("score", 1.0)),
            tags=tags
        )
    elif typ == "UserProfileComponent":
        return UserProfileComponent(
            id=id,
            name=fields.get("name", "Recovered"),
            preferences=fields.get("preferences", {"recovered": content}),
            score=fields.get("score", data.get("score", 1.0)),
            tags=tags
        )
    elif typ == "AgentGoalComponent":
        return AgentGoalComponent(
            id=id,
            goal_description=fields.get("goal_description", content),
            agent_id=fields.get("agent_id", data.get("agent_id", "unknown")),
            priority=fields.get("priority", data.get("priority", 1.0)),
            status=fields.get("status", data.get("status", "active")),
            progress=fields.get("progress", data.get("progress", 0.0)),
            deadline=fields.get("deadline", data.get("deadline")),
            created_at=fields.get("created_at"),
            last_updated=fields.get("last_updated"),
            tags=tags
        )
    elif typ == "AgentSessionComponent":
        return AgentSessionComponent(
            id=id,
            agent_id=fields.get("agent_id", data.get("agent_id", "unknown")),
            session_type=fields.get("session_type", data.get("session_type", "unknown")),
            summary=fields.get("summary", content),
            duration_minutes=fields.get("duration_minutes", data.get("duration_minutes", 0.0)),
            success=fields.get("success", data.get("success", True)),
            timestamp=fields.get("timestamp"),
            tags=tags
        )
    elif typ == "DerivedMemoryComponent":
        return DerivedMemoryComponent(
            id=id,
            content=fields.get("content", content),
            derivation=fields.get("derivation", "consolidation"),
            score=fields.get("score", data.get("score", 1.0)),
            tags=tags,
        )
    else:
        codec = _COMPONENT_CODECS.get(typ)
        return codec[1](id, fields, tags) if codec else None


def component_from_dict(id: str, data: Dict[str, Any]) -> Optional[ContextComponent]:
    """Deserialize a component and restore lifecycle metadata when present."""
    component = _deserialize_component(id, data)
    memory_data = data.get("memory")
    if memory_data is None and isinstance(data.get("metadata"), dict):
        memory_data = data["metadata"].get("memory")
    if component is not None and memory_data is not None:
        component.set_memory_lifecycle(MemoryLifecycle.from_dict(memory_data))
    return component

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
                model=summarizer_config.get("model", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")),
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
                model=summarizer_config.get("model", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")),
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
    if mem_type == "postgres_vector":
        try:
            memory_store = PostgreSQLVectorMemoryStore(
                host=mem_conf.get("host", "localhost"),
                port=mem_conf.get("port", 5432),
                database=mem_conf.get("database", "ai_context"),
                user=mem_conf.get("user", "postgres"),
                password=mem_conf.get("password", os.getenv("POSTGRES_PASSWORD", "")),  # nosec B106
                table_name=mem_conf.get("table_name", "agent_memory"),
                embedding_dimension=mem_conf.get("embedding_dimension", 384),
                embedding_model=mem_conf.get("embedding_model", "all-MiniLM-L6-v2"),
                max_connections=mem_conf.get("max_connections", 20),
                index_type=mem_conf.get("index_type", "hnsw"),
                index_parameters=mem_conf.get("index_parameters", {})
            )
            logging.info("Using PostgreSQL vector database memory store")
        except ImportError as e:
            logging.warning(f"PostgreSQL vector database not available: {e}. Falling back to JSON store.")
            memory_store = JSONMemoryStore(mem_conf.get("filepath", "memory.json"))
    elif mem_type == "vector":
        try:
            memory_store = VectorMemoryStore(
                collection_name=mem_conf.get("collection_name", "agent_memory"),
                persist_directory=mem_conf.get("persist_directory", "./chroma_db"),
                embedding_model=mem_conf.get("embedding_model", "all-MiniLM-L6-v2")
            )
            logging.info("Using ChromaDB vector memory store")
        except ImportError as e:
            logging.warning(f"Vector database not available: {e}. Falling back to JSON store.")
            memory_store = JSONMemoryStore(mem_conf.get("filepath", "memory.json"))
    elif mem_type == "json":
        memory_store = JSONMemoryStore(mem_conf.get("filepath", "memory.json"))
    elif mem_type == "sqlite":
        memory_store = SQLiteMemoryStore(mem_conf.get("db_path", "memory.db"))
    else:
        memory_store = None

    return feedback_store, memory_store
