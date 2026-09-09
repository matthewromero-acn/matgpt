from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from matgpt.client import MatGPTClient
from matgpt.context import annotate_tokens, fit_to_window
from matgpt.message import Message, Role
from matgpt.models import ModelManager

if TYPE_CHECKING:
    from matgpt.embeddings import EmbeddingClient
    from matgpt.storage import Storage

# Minimum similarity threshold — memories below this are too unrelated to inject
_MIN_SIMILARITY = 0.45
# Max number of past memories to surface per turn
_MAX_MEMORIES = 3


class Conversation:
    def __init__(
        self,
        id: str,
        name: str,
        model_manager: ModelManager,
        client: MatGPTClient,
        system_prompt: str = "",
        max_tokens: int = 4096,
        embedding_client: EmbeddingClient | None = None,
        storage: Storage | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self._model_manager = model_manager
        self._client = client
        self._max_tokens = max_tokens
        self._messages: list[Message] = []
        self._system_msg: Message | None = None
        self._embedding_client = embedding_client
        self._storage = storage
        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, prompt: str) -> None:
        """Create or replace the system message with token annotation.

        Passing an empty string clears the system message.
        """
        if not prompt:
            self._system_msg = None
            return
        msg = Message(role=Role.SYSTEM, content=prompt)
        annotated = annotate_tokens([msg])
        self._system_msg = annotated[0]

    def add_user_message(self, content: str) -> Message:
        """Annotate a user message and append it to history."""
        msg = annotate_tokens([Message(role=Role.USER, content=content)])[0]
        self._messages.append(msg)
        return msg

    def _build_payload(self, memory_block: str | None = None) -> list[Message]:
        """Annotate current messages and apply context-window trimming.

        If memory_block is provided it is prepended to the system prompt so the
        model sees relevant past context without it counting against the
        rolling message history.
        """
        annotated = annotate_tokens(self._messages)

        if memory_block and self._system_msg:
            from dataclasses import replace as dc_replace
            enhanced = dc_replace(
                self._system_msg,
                content=f"{self._system_msg.content}\n\n{memory_block}",
            )
            enhanced = annotate_tokens([enhanced])[0]
        else:
            enhanced = self._system_msg

        return fit_to_window(annotated, self._max_tokens, enhanced)

    def _retrieve_memories(self, user_input: str) -> str | None:
        """Embed user_input, search past conversations, return a formatted block."""
        if self._embedding_client is None or self._storage is None:
            return None
        try:
            query_vec = self._embedding_client.embed(user_input)
            hits = self._storage.search_similar(
                query_vec, top_k=_MAX_MEMORIES, exclude_conv_id=self.id
            )
            relevant = [h for h in hits if h["similarity"] >= _MIN_SIMILARITY]
            if not relevant:
                return None
            snippets = "\n".join(f'- {h["content"]}' for h in relevant)
            return f"[Relevant memories from past conversations]\n{snippets}"
        except Exception:
            # Embedding failures are non-fatal — degrade gracefully
            return None

    def _save_exchange_embedding(self, user_input: str, assistant_reply: str) -> None:
        """Embed the user+assistant exchange and store it for future RAG lookups."""
        if self._embedding_client is None or self._storage is None:
            return
        try:
            from matgpt.embeddings import serialize
            text = f"User: {user_input}\nAssistant: {assistant_reply}"
            vec = self._embedding_client.embed(text)
            self._storage.save_embedding(self.id, text, serialize(vec))
        except Exception:
            pass  # non-fatal

    def chat(self, user_input: str, stream: bool = False, thinking: bool = False) -> str | Iterator[str]:
        """Send a user message and return the assistant reply.

        Non-stream: returns str.
        Stream: returns Iterator[str] that appends the assistant message when exhausted.
        thinking: if True, prepends /think so Qwen3 shows its reasoning inside
                  <think>…</think> tags. The prefix is injected into the payload
                  only — it is NOT stored in the conversation history.
        """
        memory_block = self._retrieve_memories(user_input)
        self.add_user_message(user_input)

        # Build payload then inject thinking prefix into the last user message
        payload = self._build_payload(memory_block)
        if thinking and payload and payload[-1].role == Role.USER:
            from dataclasses import replace as dc_replace
            payload[-1] = dc_replace(payload[-1], content=f"/think {payload[-1].content}")
        elif not thinking and payload and payload[-1].role == Role.USER:
            from dataclasses import replace as dc_replace
            payload[-1] = dc_replace(payload[-1], content=f"/no_think {payload[-1].content}")
        model = self._model_manager.current()

        if stream:
            def _stream() -> Iterator[str]:
                full = ""
                for chunk in self._client.stream(payload, model=model):
                    full += chunk
                    yield chunk
                self._messages.append(
                    annotate_tokens([Message(role=Role.ASSISTANT, content=full)])[0]
                )
                self._save_exchange_embedding(user_input, full)
            return _stream()
        else:
            response = self._client.complete(payload, model=model)
            self._messages.append(
                annotate_tokens([Message(role=Role.ASSISTANT, content=response)])[0]
            )
            self._save_exchange_embedding(user_input, response)
            return response

    def history(self) -> list[Message]:
        """Return all messages: system prompt first (if set), then conversation messages."""
        if self._system_msg:
            return [self._system_msg] + list(self._messages)
        return list(self._messages)

    def clear(self) -> None:
        """Wipe conversation messages but keep the system prompt."""
        self._messages = []

    def token_usage(self) -> dict[str, int]:
        """Return token counts for system, messages, and total."""
        sys_tokens = (self._system_msg.token_count or 0) if self._system_msg else 0
        msg_tokens = sum(m.token_count or 0 for m in self._messages)
        return {
            "total": sys_tokens + msg_tokens,
            "system": sys_tokens,
            "messages": msg_tokens,
        }
