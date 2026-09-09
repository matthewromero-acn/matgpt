from __future__ import annotations
import uuid

from matgpt.client import MatGPTClient
from matgpt.config import Config
from matgpt.conversation import Conversation
from matgpt.message import Role
from matgpt.models import ModelManager
from matgpt.storage import Storage


class SessionManager:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = MatGPTClient(config)
        self._model_manager = ModelManager(self._client)
        self._storage = Storage(config.db_path)

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    def new_conversation(self, name: str, system_prompt: str = "") -> Conversation:
        return Conversation(
            id=str(uuid.uuid4()),
            name=name,
            model_manager=self._model_manager,
            client=self._client,
            system_prompt=system_prompt,
            max_tokens=self._config.default_context_window,
        )

    def save_conversation(self, conv: Conversation) -> None:
        self._storage.save_conversation(conv.id, conv.name, conv.history())

    def load_conversation(self, conv_id: str) -> Conversation | None:
        result = self._storage.load_conversation(conv_id)
        if result is None:
            return None
        name, messages = result

        system_msg = next((m for m in messages if m.role == Role.SYSTEM), None)
        other_msgs = [m for m in messages if m.role != Role.SYSTEM]

        conv = Conversation(
            id=conv_id,
            name=name,
            model_manager=self._model_manager,
            client=self._client,
            system_prompt=system_msg.content if system_msg else "",
            max_tokens=self._config.default_context_window,
        )
        conv._messages = other_msgs
        return conv

    def list_conversations(self) -> list[dict]:
        return self._storage.list_conversations()

    def delete_conversation(self, conv_id: str) -> None:
        self._storage.delete_conversation(conv_id)

    def export_conversation(self, conv_id: str) -> str:
        """Export a conversation as a JSON string. Raises ValueError if not found."""
        return self._storage.export_json(conv_id)
