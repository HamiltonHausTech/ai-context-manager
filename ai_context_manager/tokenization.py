"""Token estimation with lazy, offline-safe optional tiktoken support."""

import logging
from importlib import import_module
from typing import Any, Optional

_encoding: Optional[Any] = None
_encoding_loaded = False
_fallback_warning_emitted = False


def _get_encoding() -> Optional[Any]:
    global _encoding, _encoding_loaded
    if _encoding_loaded:
        return _encoding

    _encoding_loaded = True
    try:
        tiktoken = import_module("tiktoken")
        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        # tiktoken can be installed while its encoding data is unavailable
        # offline. Token estimation must remain a local, best-effort operation.
        logging.getLogger(__name__).debug(
            "Exact tokenizer unavailable; using word-count estimation: %s", exc
        )
        _encoding = None
    return _encoding


def estimate_tokens(text: str) -> int:
    """Estimate token count without requiring network access at import time."""
    global _fallback_warning_emitted
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    if not _fallback_warning_emitted:
        logging.getLogger(__name__).warning(
            "Falling back to word-count token estimation because tiktoken is unavailable."
        )
        _fallback_warning_emitted = True
    return len(text.split())


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Return text that is guaranteed not to exceed the requested token budget."""
    if max_tokens <= 0:
        return ""
    encoding = _get_encoding()
    if encoding is not None:
        token_ids = encoding.encode(text)
        return encoding.decode(token_ids[:max_tokens])
    return " ".join(text.split()[:max_tokens])
