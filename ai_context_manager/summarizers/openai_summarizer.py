import logging
import os
from openai import OpenAI
from ai_context_manager.summarizers import Summarizer

class OpenAISummarizer(Summarizer):
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def summarize(self, text: str, max_tokens: int = 100) -> str:
        prompt = f"Summarize the following in ~{max_tokens} tokens:\n\n{text}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
            logging.info(f"[summarizer:openai] model={self.model} output_preview={result[:60]!r}")
            return result
        except Exception as e:
            logging.warning(f"[summarizer:openai] failed: {e}")
            return "[summary unavailable]"
