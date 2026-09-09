import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    base_url: str
    api_key: str
    db_path: str
    default_context_window: int


def get_config() -> Config:
    default_db = str(Path.home() / ".matgpt" / "matgpt.db")
    return Config(
        base_url=os.environ.get("MATGPT_BASE_URL", "http://localhost:1234/v1"),
        api_key=os.environ.get("MATGPT_API_KEY", "lm-studio"),
        db_path=os.environ.get("MATGPT_DB_PATH", default_db),
        default_context_window=int(os.environ.get("MATGPT_CONTEXT_WINDOW", "4096")),
    )
