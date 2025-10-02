from ai_context_manager.summarizers import Summarizer

class NaiveSummarizer(Summarizer):
    def summarize(self, text: str, max_tokens: int) -> str:
        approx_chars = max_tokens * 4
        return text[:approx_chars] + " ...[truncated]"