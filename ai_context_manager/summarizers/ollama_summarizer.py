import logging
import requests
import time

from ai_context_manager.summarizers import Summarizer
# --- Ollama Summarizer ---

class OllamaSummarizer(Summarizer):
    def __init__(self, model="mistral", host="http://192.168.0.156:11434",timeout=30):
        self.model = model
        self.api_url = f"{host}/api/generate"
        self.timeout = timeout

    def summarize(self, text: str, max_tokens: int = 100) -> str:
        """Summarize text using Ollama with comprehensive error handling."""
        if not text or not text.strip():
            logging.warning("Empty text provided for summarization")
            return ""
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        prompt = f"Summarize the following in ~{max_tokens} tokens:\n\n{text}"
        start = time.time()
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            summary = response_data.get("response", "").strip()
            
            if not summary:
                logging.warning("Received empty summary from Ollama")
                return "[summary unavailable - empty response]"
            
            duration = time.time() - start
            logging.info(f"[summarizer:ollama] model={self.model} input_chars={len(text)} "
                         f"duration={duration:.2f}s output_preview={summary[:60]!r}")
            return summary
            
        except requests.exceptions.Timeout:
            duration = time.time() - start
            logging.error(f"[summarizer:ollama] timeout after {duration:.2f}s (limit: {self.timeout}s)")
            return "[summary unavailable - timeout]"
        except requests.exceptions.ConnectionError as e:
            duration = time.time() - start
            logging.error(f"[summarizer:ollama] connection error after {duration:.2f}s: {e}")
            return "[summary unavailable - connection error]"
        except requests.exceptions.HTTPError as e:
            duration = time.time() - start
            logging.error(f"[summarizer:ollama] HTTP error after {duration:.2f}s: {e}")
            return "[summary unavailable - HTTP error]"
        except ValueError as e:
            duration = time.time() - start
            logging.error(f"[summarizer:ollama] JSON decode error after {duration:.2f}s: {e}")
            return "[summary unavailable - invalid response]"
        except Exception as e:
            duration = time.time() - start
            logging.error(f"[summarizer:ollama] unexpected error after {duration:.2f}s: {e}")
            return "[summary unavailable - unexpected error]"
