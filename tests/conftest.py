import pytest
from unittest.mock import MagicMock
from matgpt.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        db_path=str(tmp_path / "test.db"),
        default_context_window=4096,
    )
