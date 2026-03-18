from __future__ import annotations


def truncate_prompt_text(text: str, *, max_chars: int = 100_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."
