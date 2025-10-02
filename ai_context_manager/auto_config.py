"""
Automatic configuration detection and setup
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AutoConfig:
    """Automatic configuration detection and setup."""
    
    def __init__(self):
        self.detected_config = {}
        
    def detect_environment(self) -> Dict[str, Any]:
        """Auto-detect environment and create appropriate configuration."""
        config = {
            "logging": {"level": "INFO", "format": "[%(levelname)s] %(message)s"},
            "feedback_store": {"type": "json", "filepath": "feedback.json"},
            "memory_store": {"type": "json", "filepath": "memory.json"},
            "summarizer": {"type": "naive", "model": "gpt-3.5-turbo"}
        }
        
        # Detect available dependencies
        self._detect_summarizer(config)
        self._detect_storage(config)
        self._detect_network(config)
        
        self.detected_config = config
        return config
    
    def _detect_summarizer(self, config: Dict[str, Any]):
        """Detect available summarizer options."""
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            config["summarizer"]["type"] = "openai"
            config["summarizer"]["model"] = "gpt-3.5-turbo"
            logger.info("Detected OpenAI API key - using OpenAI summarizer")
            return
        
        # Check for Ollama
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if self._test_ollama_connection(ollama_host):
            config["summarizer"]["type"] = "ollama"
            config["summarizer"]["model"] = os.getenv("OLLAMA_MODEL", "mistral")
            config["summarizer"]["host"] = ollama_host
            logger.info(f"Detected Ollama at {ollama_host} - using Ollama summarizer")
            return
        
        # Default to auto-fallback
        config["summarizer"]["type"] = "auto_fallback"
        config["summarizer"]["model"] = "mistral"
        logger.info("Using auto-fallback summarizer")
    
    def _detect_storage(self, config: Dict[str, Any]):
        """Detect available storage options."""
        # Check for vector database dependencies
        try:
            import chromadb
            import sentence_transformers
            config["memory_store"]["type"] = "vector"
            config["memory_store"]["collection_name"] = "agent_memory"
            config["memory_store"]["persist_directory"] = "./chroma_db"
            config["memory_store"]["embedding_model"] = "all-MiniLM-L6-v2"
            logger.info("Detected vector database dependencies - using ChromaDB")
            return
        except ImportError:
            pass
        
        # Check for SQLite preference
        if os.getenv("USE_SQLITE", "").lower() == "true":
            config["memory_store"]["type"] = "sqlite"
            config["memory_store"]["db_path"] = "context.db"
            logger.info("Using SQLite storage")
            return
        
        # Default to JSON
        logger.info("Using JSON storage")
    
    def _detect_network(self, config: Dict[str, Any]):
        """Detect network environment."""
        # Check if we're in a containerized environment
        if os.path.exists("/.dockerenv") or os.getenv("CONTAINER"):
            config["summarizer"]["host"] = "http://host.docker.internal:11434"
            logger.info("Detected containerized environment")
        
        # Check for local network Ollama
        if os.getenv("OLLAMA_HOST"):
            config["summarizer"]["host"] = os.getenv("OLLAMA_HOST")
    
    def _test_ollama_connection(self, host: str) -> bool:
        """Test if Ollama is available."""
        try:
            import requests
            response = requests.get(f"{host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def create_config_file(self, path: str = "config.toml"):
        """Create configuration file with detected settings."""
        import toml
        
        with open(path, 'w') as f:
            toml.dump(self.detected_config, f)
        
        logger.info(f"Created configuration file: {path}")
    
    def get_recommended_config(self, use_case: str = "agent") -> Dict[str, Any]:
        """Get recommended configuration for specific use case."""
        base_config = self.detect_environment()
        
        if use_case == "agent":
            # Optimize for long-running agents
            base_config["memory_store"]["type"] = "vector"
            base_config["summarizer"]["type"] = "auto_fallback"
        elif use_case == "simple":
            # Optimize for simple usage
            base_config["memory_store"]["type"] = "json"
            base_config["summarizer"]["type"] = "naive"
        elif use_case == "production":
            # Optimize for production
            base_config["memory_store"]["type"] = "sqlite"
            base_config["summarizer"]["type"] = "openai"
        
        return base_config


def auto_detect_and_setup() -> Dict[str, Any]:
    """Auto-detect environment and return optimal configuration."""
    auto_config = AutoConfig()
    return auto_config.detect_environment()


def create_optimal_config(use_case: str = "agent") -> Dict[str, Any]:
    """Create optimal configuration for specific use case."""
    auto_config = AutoConfig()
    return auto_config.get_recommended_config(use_case)


def setup_with_auto_config(use_case: str = "agent") -> Dict[str, Any]:
    """Setup system with automatically detected optimal configuration."""
    config = create_optimal_config(use_case)
    
    # Create config file
    auto_config = AutoConfig()
    auto_config.detected_config = config
    auto_config.create_config_file()
    
    return config
