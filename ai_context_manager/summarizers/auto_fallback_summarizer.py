import logging
import time
from typing import Optional

from .base import Summarizer
from .naive_summarizer import NaiveSummarizer
from .ollama_summarizer import OllamaSummarizer

logger = logging.getLogger(__name__)

class AutoFallbackSummarizer(Summarizer):
    """
    A summarizer that automatically tries Ollama first, then falls back to naive summarizer.
    Tests Ollama connectivity on first use and caches the result.
    """
    
    def __init__(self, model="mistral", host="http://localhost:11434", timeout=10, 
                 health_check_timeout=5, cache_health_result=True):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout
        self.cache_health_result = cache_health_result
        
        # Initialize summarizers
        self.ollama_summarizer = OllamaSummarizer(model=model, host=host, timeout=timeout)
        self.naive_summarizer = NaiveSummarizer()
        
        # Health check state
        self._ollama_available = None  # None = not tested, True = available, False = unavailable
        self._last_health_check = 0
        self._health_check_interval = 300  # Re-check every 5 minutes
        
    def _is_ollama_available(self) -> bool:
        """Check if Ollama is available, with caching and periodic re-checking."""
        now = time.time()
        
        # If we have a cached result and it's recent enough, use it
        if (self._ollama_available is not None and 
            self.cache_health_result and 
            (now - self._last_health_check) < self._health_check_interval):
            return self._ollama_available
        
        # Perform health check
        try:
            logger.debug(f"Testing Ollama connectivity to {self.host}")
            
            # Try a simple API call to test connectivity
            import requests
            health_url = f"{self.host}/api/tags"
            response = requests.get(health_url, timeout=self.health_check_timeout)
            response.raise_for_status()
            
            # If we get here, Ollama is available
            self._ollama_available = True
            self._last_health_check = now
            logger.info(f"Ollama is available at {self.host}")
            return True
            
        except Exception as e:
            self._ollama_available = False
            self._last_health_check = now
            logger.debug(f"Ollama not available at {self.host}: {e}")
            return False
    
    def summarize(self, text: str, max_tokens: int = 100) -> str:
        """Summarize text using Ollama if available, otherwise fall back to naive summarizer."""
        if not text or not text.strip():
            logger.warning("Empty text provided for summarization")
            return ""
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Try Ollama first if available
        if self._is_ollama_available():
            try:
                logger.debug(f"Using Ollama summarizer for {len(text)} characters")
                result = self.ollama_summarizer.summarize(text, max_tokens)
                
                # If Ollama returned a valid result (not an error message), use it
                if result and not result.startswith("[summary unavailable"):
                    return result
                else:
                    logger.warning("Ollama returned error, falling back to naive summarizer")
                    
            except Exception as e:
                logger.warning(f"Ollama summarization failed: {e}, falling back to naive summarizer")
                # Mark as unavailable for this session to avoid repeated failures
                self._ollama_available = False
        
        # Fall back to naive summarizer
        logger.debug(f"Using naive summarizer for {len(text)} characters")
        return self.naive_summarizer.summarize(text, max_tokens)
    
    def get_status(self) -> dict:
        """Get current status of the summarizer."""
        ollama_available = self._is_ollama_available()
        return {
            "type": "auto_fallback",
            "ollama_available": ollama_available,
            "ollama_host": self.host,
            "ollama_model": self.model,
            "last_health_check": self._last_health_check,
            "fallback_summarizer": "naive"
        }
    
    def force_health_check(self) -> bool:
        """Force a new health check of Ollama availability."""
        self._ollama_available = None
        self._last_health_check = 0
        return self._is_ollama_available()
