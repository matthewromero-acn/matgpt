import os
import pytest
from matgpt.config import Config, get_config


def test_get_config_defaults():
    # Clear any env overrides
    os.environ.pop("MATGPT_BASE_URL", None)
    os.environ.pop("MATGPT_DB_PATH", None)
    cfg = get_config()
    assert cfg.base_url == "http://localhost:1234/v1"
    assert cfg.api_key == "lm-studio"
    assert cfg.db_path.endswith("matgpt.db")
    assert cfg.default_context_window == 4096


def test_get_config_env_override(monkeypatch):
    monkeypatch.setenv("MATGPT_BASE_URL", "http://remote:1234/v1")
    monkeypatch.setenv("MATGPT_DB_PATH", "/tmp/test.db")
    cfg = get_config()
    assert cfg.base_url == "http://remote:1234/v1"
    assert cfg.db_path == "/tmp/test.db"
