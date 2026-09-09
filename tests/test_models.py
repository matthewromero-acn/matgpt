import pytest
from unittest.mock import MagicMock
from matgpt.models import ModelManager


def make_client(model_ids: list[str]) -> MagicMock:
    client = MagicMock()
    client.list_models.return_value = model_ids
    return client


def test_available_returns_models():
    client = make_client(["llama-3", "mistral-7b"])
    mgr = ModelManager(client)
    assert mgr.available() == ["llama-3", "mistral-7b"]


def test_available_is_cached():
    client = make_client(["llama-3"])
    mgr = ModelManager(client)
    mgr.available()
    mgr.available()
    assert client.list_models.call_count == 1  # fetched once, then cached


def test_select_valid_model():
    client = make_client(["llama-3", "mistral-7b"])
    mgr = ModelManager(client)
    mgr.select("llama-3")
    assert mgr.current() == "llama-3"


def test_select_invalid_model_raises():
    client = make_client(["llama-3"])
    mgr = ModelManager(client)
    with pytest.raises(ValueError, match="not available"):
        mgr.select("gpt-99")


def test_current_raises_when_none_selected():
    client = make_client(["llama-3"])
    mgr = ModelManager(client)
    with pytest.raises(RuntimeError, match="No model selected"):
        mgr.current()


def test_refresh_clears_cache():
    client = make_client(["llama-3"])
    mgr = ModelManager(client)
    mgr.available()
    client.list_models.return_value = ["llama-3", "phi-3"]
    updated = mgr.refresh()
    assert updated == ["llama-3", "phi-3"]
    assert client.list_models.call_count == 2
