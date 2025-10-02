import toml
import os
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# --- Config Class ---
class Config:
    def __init__(self, path="config.toml"):
        self.data = toml.load(path)
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration sections and values."""
        required_sections = ["logging", "summarizer", "feedback_store", "memory_store"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate summarizer config
        summarizer_config = self.data.get("summarizer", {})
        if summarizer_config.get("type") == "openai":
            api_key = self.get_api_key()
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable or config file (deprecated)."""
        # Prioritize environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # Fallback to config file (for backward compatibility, but warn)
        api_key = self.data.get("summarizer", {}).get("api_key")
        if api_key:
            logger.warning("Using API key from config file. Consider using OPENAI_API_KEY environment variable instead.")
            return api_key
        
        return None

    def load_config(self, section: str, default: Any = None) -> Any:
        """Load an entire configuration section."""
        return self.data.get(section, default)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        return self.data.get(section, {}).get(key, default)
    
    def get_with_env_fallback(self, section: str, key: str, env_var: str, default: Any = None) -> Any:
        """Get config value with environment variable fallback."""
        # Check environment variable first
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value
        
        # Fallback to config file
        return self.get(section, key, default)