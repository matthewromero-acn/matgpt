from __future__ import annotations
from collections.abc import Iterator
from openai import OpenAI

from matgpt.config import Config
from matgpt.message import Message


class MatGPTClient:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._openai = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def list_models(self) -> list[str]:
        response = self._openai.models.list()
        return [m.id for m in response.data]

    def complete(self, messages: list[Message], model: str, **kwargs) -> str:
        payload = [m.to_dict() for m in messages]
        response = self._openai.chat.completions.create(
            model=model,
            messages=payload,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: list[Message], model: str, **kwargs) -> Iterator[str]:
        payload = [m.to_dict() for m in messages]
        response = self._openai.chat.completions.create(
            model=model,
            messages=payload,
            stream=True,
            **kwargs,
        )
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            response.close()
