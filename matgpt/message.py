from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int | None = None

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}
