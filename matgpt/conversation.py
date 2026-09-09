from __future__ import annotations

from collections.abc import Iterator

from matgpt.client import MatGPTClient
from matgpt.context import annotate_tokens, fit_to_window
from matgpt.message import Message, Role
from matgpt.models import ModelManager


class Conversation:
    def __init__(
        self,
        id: str,
        name: str,
        model_manager: ModelManager,
        client: MatGPTClient,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> None:
        self.id = id
        self.name = name
        self._model_manager = model_manager
        self._client = client
        self._max_tokens = max_tokens
        self._messages: list[Message] = []
        self._system_msg: Message | None = None
        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, prompt: str) -> None:
        """Create or replace the system message with token annotation."""
        msg = Message(role=Role.SYSTEM, content=prompt)
        annotated = annotate_tokens([msg])
        self._system_msg = annotated[0]

    def add_user_message(self, content: str) -> Message:
        """Annotate a user message and append it to history."""
        msg = annotate_tokens([Message(role=Role.USER, content=content)])[0]
        self._messages.append(msg)
        return msg

    def _build_payload(self) -> list[Message]:
        """Annotate current messages and apply context-window trimming."""
        annotated = annotate_tokens(self._messages)
        return fit_to_window(annotated, self._max_tokens, self._system_msg)

    def chat(self, user_input: str, stream: bool = False) -> str | Iterator[str]:
        """Send a user message and return the assistant reply.

        Non-stream: returns str.
        Stream: returns Iterator[str] that appends the assistant message when exhausted.
        """
        self.add_user_message(user_input)
        payload = self._build_payload()
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
            return _stream()
        else:
            response = self._client.complete(payload, model=model)
            self._messages.append(
                annotate_tokens([Message(role=Role.ASSISTANT, content=response)])[0]
            )
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
        sys_tokens = self._system_msg.token_count or 0 if self._system_msg else 0
        msg_tokens = sum(m.token_count or 0 for m in self._messages)
        return {
            "total": sys_tokens + msg_tokens,
            "system": sys_tokens,
            "messages": msg_tokens,
        }
