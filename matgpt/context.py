from __future__ import annotations

from dataclasses import replace

import tiktoken

from matgpt.message import Message, Role

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def annotate_tokens(messages: list[Message]) -> list[Message]:
    """Return new list of Messages with token_count filled in. Originals unchanged."""
    result = []
    for msg in messages:
        new_msg = replace(msg, token_count=count_tokens(msg.content))
        result.append(new_msg)
    return result


def fit_to_window(
    messages: list[Message],
    max_tokens: int,
    system_msg: Message | None = None,
) -> list[Message]:
    """
    Trim messages to fit within max_tokens.
    - system_msg is always kept first if provided.
    - Oldest messages are dropped first.
    - All messages must have token_count set (run annotate_tokens first).
    """
    if system_msg is not None:
        sys_tokens = system_msg.token_count or count_tokens(system_msg.content)
        if sys_tokens > max_tokens:
            raise ValueError(
                f"System prompt alone ({sys_tokens} tokens) exceeds max_tokens ({max_tokens})."
            )
        budget = max_tokens - sys_tokens
    else:
        budget = max_tokens

    # Walk from newest to oldest, keep as many as fit
    kept: list[Message] = []
    for msg in reversed(messages):
        tokens = msg.token_count or count_tokens(msg.content)
        if tokens <= budget:
            kept.append(msg)
            budget -= tokens
        # If it doesn't fit, drop it (oldest messages fall off naturally)

    kept.reverse()

    if system_msg is not None:
        return [system_msg] + kept
    return kept
