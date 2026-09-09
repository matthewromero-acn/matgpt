from __future__ import annotations
from matgpt.client import MatGPTClient


class ModelManager:
    def __init__(self, client: MatGPTClient) -> None:
        self._client = client
        self._cache: list[str] | None = None
        self._current: str | None = None

    def available(self) -> list[str]:
        if self._cache is None:
            self._cache = self._client.list_models()
        return self._cache

    def refresh(self) -> list[str]:
        self._cache = self._client.list_models()
        return self._cache

    def select(self, model_id: str) -> None:
        if model_id not in self.available():
            raise ValueError(f"Model '{model_id}' not available. Run refresh() if recently loaded.")
        self._current = model_id

    def current(self) -> str:
        if self._current is None:
            raise RuntimeError("No model selected. Call select() first.")
        return self._current
