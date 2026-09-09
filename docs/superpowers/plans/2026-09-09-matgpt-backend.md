# MatGPT Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality Python backend for MatGPT — a local-model chat system with multi-turn memory, session management, context-window optimization, and persistent conversation history.

**Architecture:** A layered Python package where a thin `client` wraps the OpenAI-compatible LM Studio API, `conversation` models messages and context, `session` manages multiple named chats, and `storage` persists everything to SQLite. Each layer has a clean interface so the UI (added later) can plug in at any level.

**Tech Stack:** Python 3.11+, `openai` SDK (≥1.0), `tiktoken` (token approximation), `sqlite3` (stdlib, no ORM), `pytest` + `pytest-asyncio`, `dataclasses`, `pydantic` (validation).

**Spec:** This document is the spec.

## Global Constraints

- Python ≥ 3.11
- `openai` SDK ≥ 1.0 — use `client.chat.completions.create` and `.stream` only (no deprecated methods)
- LM Studio server assumed at `http://localhost:1234/v1` by default; configurable via env var `MATGPT_BASE_URL`
- All public functions must have type annotations
- No async/await — synchronous only for now (simpler, easier to test, UI layer can thread as needed)
- SQLite DB file defaults to `~/.matgpt/matgpt.db`; configurable via env var `MATGPT_DB_PATH`
- Token counting is approximate (tiktoken `cl100k_base` encoding used as proxy for all local models)
- Max context window defaults to 4096 tokens; configurable per model

---

## File Map

```
matgpt/
├── matgpt/
│   ├── __init__.py          # Public API surface
│   ├── config.py            # Settings dataclass, env var loading
│   ├── client.py            # LM Studio HTTP client wrapper
│   ├── models.py            # Model listing, loading, selection
│   ├── message.py           # Message dataclass + role enum
│   ├── conversation.py      # Conversation class: history, system prompt, token tracking
│   ├── context.py           # Context window management: truncation strategy
│   ├── session.py           # SessionManager: multiple named conversations
│   └── storage.py           # SQLite persistence: save/load conversations
├── tests/
│   ├── conftest.py          # Shared fixtures (mock client, tmp db)
│   ├── test_config.py
│   ├── test_client.py
│   ├── test_models.py
│   ├── test_message.py
│   ├── test_conversation.py
│   ├── test_context.py
│   ├── test_session.py
│   └── test_storage.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Task 1: Project Scaffold + Config

**Files:**
- Create: `matgpt/__init__.py`
- Create: `matgpt/config.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` dataclass with fields `base_url: str`, `api_key: str`, `db_path: str`, `default_context_window: int`
- Produces: `get_config() -> Config` — reads env vars, falls back to defaults

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "matgpt"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0.0",
    "tiktoken>=0.7.0",
    "pydantic>=2.0.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["matgpt*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `requirements.txt`**

```
openai>=1.0.0
tiktoken>=0.7.0
pydantic>=2.0.0
pytest>=8.0.0
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt
```

- [ ] **Step 6: Implement `matgpt/config.py`**

```python
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
```

- [ ] **Step 7: Create `matgpt/__init__.py`**

```python
from matgpt.config import Config, get_config

__all__ = ["Config", "get_config"]
```

- [ ] **Step 8: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 9: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: 2 PASSED

- [ ] **Step 10: Commit**

```bash
git init
git add .
git commit -m "feat: project scaffold and config"
```

---

## Task 2: Message Model

**Files:**
- Create: `matgpt/message.py`
- Test: `tests/test_message.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Role` enum: `SYSTEM = "system"`, `USER = "user"`, `ASSISTANT = "assistant"`
  - `Message` dataclass: `role: Role`, `content: str`, `timestamp: datetime`, `token_count: int | None = None`
  - `Message.to_dict() -> dict[str, str]` — returns `{"role": ..., "content": ...}` for API calls

- [ ] **Step 1: Write the failing test**

```python
# tests/test_message.py
from datetime import datetime
from matgpt.message import Message, Role

def test_message_creation():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"
    assert isinstance(msg.timestamp, datetime)
    assert msg.token_count is None

def test_message_to_dict():
    msg = Message(role=Role.ASSISTANT, content="Hi there")
    d = msg.to_dict()
    assert d == {"role": "assistant", "content": "Hi there"}

def test_role_values():
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_message.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/message.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int | None = None

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_message.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/message.py tests/test_message.py
git commit -m "feat: add Message model and Role enum"
```

---

## Task 3: LM Studio Client Wrapper

**Files:**
- Create: `matgpt/client.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `Config` from `matgpt.config`; `Message` from `matgpt.message`
- Produces:
  - `MatGPTClient(config: Config)`
  - `MatGPTClient.complete(messages: list[Message], model: str, **kwargs) -> str` — returns full response text
  - `MatGPTClient.stream(messages: list[Message], model: str, **kwargs) -> Iterator[str]` — yields text chunks
  - `MatGPTClient.list_models() -> list[str]` — returns model IDs

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_client.py
from unittest.mock import MagicMock, patch, PropertyMock
from matgpt.client import MatGPTClient
from matgpt.message import Message, Role


def test_complete_returns_string(config):
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "42"
    mock_openai.chat.completions.create.return_value.choices = [mock_choice]

    with patch("matgpt.client.OpenAI", return_value=mock_openai):
        client = MatGPTClient(config)
        messages = [Message(role=Role.USER, content="What is 2+2?")]
        result = client.complete(messages, model="test-model")

    assert result == "42"


def test_list_models_returns_ids(config):
    mock_openai = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "llama-3"
    mock_openai.models.list.return_value.data = [mock_model]

    with patch("matgpt.client.OpenAI", return_value=mock_openai):
        client = MatGPTClient(config)
        models = client.list_models()

    assert models == ["llama-3"]


def test_stream_yields_chunks(config):
    mock_openai = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "Hello"
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = " world"

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__ = MagicMock(return_value=iter([mock_chunk1, mock_chunk2]))
    mock_stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_openai.chat.completions.stream.return_value = mock_stream_ctx

    with patch("matgpt.client.OpenAI", return_value=mock_openai):
        client = MatGPTClient(config)
        messages = [Message(role=Role.USER, content="Hi")]
        chunks = list(client.stream(messages, model="test-model"))

    assert chunks == ["Hello", " world"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_client.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/client.py`**

```python
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
        with self._openai.chat.completions.stream(
            model=model,
            messages=payload,
            **kwargs,
        ) as s:
            for chunk in s:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_client.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/client.py tests/test_client.py
git commit -m "feat: add MatGPTClient with complete, stream, list_models"
```

---

## Task 4: Model Manager

**Files:**
- Create: `matgpt/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `MatGPTClient` from `matgpt.client`
- Produces:
  - `ModelManager(client: MatGPTClient)`
  - `ModelManager.available() -> list[str]` — cached list of model IDs
  - `ModelManager.select(model_id: str) -> None` — set active model, raises `ValueError` if not in available list
  - `ModelManager.current() -> str` — returns active model ID, raises `RuntimeError` if none selected
  - `ModelManager.refresh() -> list[str]` — force-refresh cache from server

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/models.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/models.py tests/test_models.py
git commit -m "feat: add ModelManager with selection and caching"
```

---

## Task 5: Context Window Manager

**Files:**
- Create: `matgpt/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `Message`, `Role` from `matgpt.message`
- Produces:
  - `count_tokens(text: str) -> int` — approximate token count via tiktoken `cl100k_base`
  - `annotate_tokens(messages: list[Message]) -> list[Message]` — returns new list with `token_count` filled in
  - `fit_to_window(messages: list[Message], max_tokens: int, system_msg: Message | None) -> list[Message]`
    - Always keeps `system_msg` first (if provided)
    - Drops oldest non-system messages until total fits within `max_tokens`
    - Raises `ValueError` if system prompt alone exceeds `max_tokens`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_context.py
import pytest
from matgpt.message import Message, Role
from matgpt.context import count_tokens, annotate_tokens, fit_to_window


def make_msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


def test_count_tokens_returns_int():
    n = count_tokens("Hello, world!")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_longer_text_is_larger():
    short = count_tokens("Hi")
    long = count_tokens("Hi " * 100)
    assert long > short


def test_annotate_tokens_fills_token_count():
    messages = [
        make_msg(Role.USER, "Hello"),
        make_msg(Role.ASSISTANT, "Hi there"),
    ]
    annotated = annotate_tokens(messages)
    assert all(m.token_count is not None and m.token_count > 0 for m in annotated)


def test_annotate_tokens_does_not_mutate_originals():
    msg = make_msg(Role.USER, "Hello")
    annotated = annotate_tokens([msg])
    assert msg.token_count is None  # original unchanged
    assert annotated[0].token_count is not None


def test_fit_to_window_keeps_system_and_recent():
    system = make_msg(Role.SYSTEM, "You are helpful.")
    # Create many messages that won't all fit
    messages = [make_msg(Role.USER, f"Message {i}" * 20) for i in range(20)]
    messages = annotate_tokens(messages)
    system = annotate_tokens([system])[0]

    result = fit_to_window(messages, max_tokens=200, system_msg=system)
    assert result[0].role == Role.SYSTEM
    assert len(result) < len(messages) + 1  # some dropped
    total = sum(m.token_count for m in result)
    assert total <= 200


def test_fit_to_window_no_system():
    messages = [make_msg(Role.USER, f"msg {i}" * 50) for i in range(10)]
    messages = annotate_tokens(messages)
    result = fit_to_window(messages, max_tokens=100, system_msg=None)
    total = sum(m.token_count for m in result)
    assert total <= 100


def test_fit_to_window_raises_if_system_too_large():
    system = make_msg(Role.SYSTEM, "word " * 500)
    system = annotate_tokens([system])[0]
    with pytest.raises(ValueError, match="System prompt"):
        fit_to_window([], max_tokens=10, system_msg=system)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_context.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/context.py`**

```python
from __future__ import annotations
import copy
import tiktoken
from matgpt.message import Message, Role

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def annotate_tokens(messages: list[Message]) -> list[Message]:
    """Return new list of Messages with token_count filled in. Originals unchanged."""
    result = []
    for msg in messages:
        new_msg = copy.replace(msg, token_count=count_tokens(msg.content))
        result.append(new_msg)
    return result


def fit_to_window(
    messages: list[Message],
    max_tokens: int,
    system_msg: Message | None = None,
) -> list[Message]:
    """
    Trim messages to fit within max_tokens.
    - system_msg is always kept first if provided.
    - Oldest messages are dropped first.
    - All messages must have token_count set (run annotate_tokens first).
    """
    if system_msg is not None:
        sys_tokens = system_msg.token_count or count_tokens(system_msg.content)
        if sys_tokens > max_tokens:
            raise ValueError(
                f"System prompt alone ({sys_tokens} tokens) exceeds max_tokens ({max_tokens})."
            )
        budget = max_tokens - sys_tokens
    else:
        budget = max_tokens

    # Walk from newest to oldest, keep as many as fit
    kept: list[Message] = []
    for msg in reversed(messages):
        tokens = msg.token_count or count_tokens(msg.content)
        if tokens <= budget:
            kept.append(msg)
            budget -= tokens
        # If it doesn't fit, drop it (oldest messages fall off naturally)

    kept.reverse()

    if system_msg is not None:
        return [system_msg] + kept
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_context.py -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/context.py tests/test_context.py
git commit -m "feat: add context window manager with token counting and truncation"
```

---

## Task 6: Conversation

**Files:**
- Create: `matgpt/conversation.py`
- Test: `tests/test_conversation.py`

**Interfaces:**
- Consumes: `Message`, `Role` from `matgpt.message`; `MatGPTClient` from `matgpt.client`; `ModelManager` from `matgpt.models`; `annotate_tokens`, `fit_to_window` from `matgpt.context`
- Produces:
  - `Conversation(id: str, name: str, model_manager: ModelManager, client: MatGPTClient, system_prompt: str = "", max_tokens: int = 4096)`
  - `Conversation.set_system_prompt(prompt: str) -> None`
  - `Conversation.add_user_message(content: str) -> Message`
  - `Conversation.chat(user_input: str, stream: bool = False) -> str | Iterator[str]`
    - Adds user message, calls model, appends assistant response, returns it
    - Applies `fit_to_window` before every API call
  - `Conversation.history() -> list[Message]` — all messages including system prompt
  - `Conversation.clear() -> None` — wipe messages but keep system prompt and settings
  - `Conversation.token_usage() -> dict[str, int]` — `{"total": N, "messages": N, "system": N}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversation.py
from unittest.mock import MagicMock
from matgpt.conversation import Conversation
from matgpt.message import Role


def make_conversation(system_prompt: str = "You are helpful.", max_tokens: int = 4096):
    client = MagicMock()
    client.complete.return_value = "I am the assistant."
    model_mgr = MagicMock()
    model_mgr.current.return_value = "test-model"
    return Conversation(
        id="conv-1",
        name="Test Chat",
        model_manager=model_mgr,
        client=client,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    ), client, model_mgr


def test_chat_adds_user_and_assistant_messages():
    conv, client, _ = make_conversation()
    result = conv.chat("Hello")
    assert result == "I am the assistant."
    history = conv.history()
    roles = [m.role for m in history]
    assert Role.SYSTEM in roles
    assert Role.USER in roles
    assert Role.ASSISTANT in roles


def test_chat_calls_client_complete(  ):
    conv, client, _ = make_conversation()
    conv.chat("Tell me a joke")
    assert client.complete.called


def test_clear_removes_messages_keeps_system():
    conv, _, _ = make_conversation(system_prompt="Be concise.")
    conv.chat("Hello")
    conv.clear()
    history = conv.history()
    assert len(history) == 1
    assert history[0].role == Role.SYSTEM
    assert history[0].content == "Be concise."


def test_token_usage_returns_dict():
    conv, _, _ = make_conversation()
    conv.chat("Hello")
    usage = conv.token_usage()
    assert "total" in usage
    assert "messages" in usage
    assert "system" in usage
    assert usage["total"] == usage["messages"] + usage["system"]


def test_set_system_prompt_updates():
    conv, _, _ = make_conversation(system_prompt="Old prompt.")
    conv.set_system_prompt("New prompt.")
    history = conv.history()
    assert history[0].content == "New prompt."


def test_no_system_prompt_works():
    conv, client, _ = make_conversation(system_prompt="")
    result = conv.chat("Hi")
    assert result == "I am the assistant."
    history = conv.history()
    assert all(m.role != Role.SYSTEM for m in history)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_conversation.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/conversation.py`**

```python
from __future__ import annotations
import uuid
from collections.abc import Iterator

from matgpt.client import MatGPTClient
from matgpt.context import annotate_tokens, fit_to_window, count_tokens
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
        msg = Message(role=Role.SYSTEM, content=prompt)
        annotated = annotate_tokens([msg])
        self._system_msg = annotated[0]

    def add_user_message(self, content: str) -> Message:
        msg = annotate_tokens([Message(role=Role.USER, content=content)])[0]
        self._messages.append(msg)
        return msg

    def _build_payload(self) -> list[Message]:
        annotated = annotate_tokens(self._messages)
        return fit_to_window(annotated, self._max_tokens, self._system_msg)

    def chat(self, user_input: str, stream: bool = False) -> str | Iterator[str]:
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
        if self._system_msg:
            return [self._system_msg] + self._messages
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def token_usage(self) -> dict[str, int]:
        sys_tokens = self._system_msg.token_count or 0 if self._system_msg else 0
        msg_tokens = sum(m.token_count or 0 for m in self._messages)
        return {
            "total": sys_tokens + msg_tokens,
            "system": sys_tokens,
            "messages": msg_tokens,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_conversation.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/conversation.py tests/test_conversation.py
git commit -m "feat: add Conversation with chat, history, context management"
```

---

## Task 7: Storage (SQLite Persistence)

**Files:**
- Create: `matgpt/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `Message`, `Role` from `matgpt.message`; `Config` from `matgpt.config`
- Produces:
  - `Storage(db_path: str)` — creates DB and tables on init
  - `Storage.save_conversation(conv_id: str, name: str, messages: list[Message]) -> None` — upsert
  - `Storage.load_conversation(conv_id: str) -> tuple[str, list[Message]] | None` — returns `(name, messages)` or `None`
  - `Storage.list_conversations() -> list[dict]` — `[{"id": ..., "name": ..., "message_count": ..., "updated_at": ...}]`
  - `Storage.delete_conversation(conv_id: str) -> None`
  - `Storage.export_json(conv_id: str) -> str` — JSON string of the conversation

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_storage.py
import json
import pytest
from datetime import datetime
from matgpt.storage import Storage
from matgpt.message import Message, Role


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "test.db"))


def make_messages() -> list[Message]:
    return [
        Message(role=Role.USER, content="Hello", token_count=2),
        Message(role=Role.ASSISTANT, content="Hi there", token_count=4),
    ]


def test_save_and_load_conversation(storage):
    messages = make_messages()
    storage.save_conversation("conv-1", "My Chat", messages)
    name, loaded = storage.load_conversation("conv-1")
    assert name == "My Chat"
    assert len(loaded) == 2
    assert loaded[0].role == Role.USER
    assert loaded[0].content == "Hello"
    assert loaded[1].role == Role.ASSISTANT


def test_load_nonexistent_returns_none(storage):
    result = storage.load_conversation("does-not-exist")
    assert result is None


def test_list_conversations(storage):
    storage.save_conversation("conv-1", "Chat A", make_messages())
    storage.save_conversation("conv-2", "Chat B", make_messages()[:1])
    listing = storage.list_conversations()
    assert len(listing) == 2
    ids = {c["id"] for c in listing}
    assert "conv-1" in ids
    assert "conv-2" in ids


def test_list_includes_message_count(storage):
    storage.save_conversation("conv-1", "Chat A", make_messages())
    listing = storage.list_conversations()
    conv = next(c for c in listing if c["id"] == "conv-1")
    assert conv["message_count"] == 2


def test_delete_conversation(storage):
    storage.save_conversation("conv-1", "Chat", make_messages())
    storage.delete_conversation("conv-1")
    assert storage.load_conversation("conv-1") is None


def test_save_is_upsert(storage):
    storage.save_conversation("conv-1", "Old Name", make_messages())
    new_msgs = [Message(role=Role.USER, content="Updated", token_count=1)]
    storage.save_conversation("conv-1", "New Name", new_msgs)
    name, loaded = storage.load_conversation("conv-1")
    assert name == "New Name"
    assert len(loaded) == 1
    assert loaded[0].content == "Updated"


def test_export_json(storage):
    storage.save_conversation("conv-1", "Chat", make_messages())
    raw = storage.export_json("conv-1")
    data = json.loads(raw)
    assert data["id"] == "conv-1"
    assert data["name"] == "Chat"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_storage.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/storage.py`**

```python
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from matgpt.message import Message, Role


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    timestamp TEXT NOT NULL,
    position INTEGER NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_conversation(self, conv_id: str, name: str, messages: list[Message]) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (id, name, updated_at) VALUES (?, ?, ?)",
                (conv_id, name, now),
            )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.executemany(
                "INSERT INTO messages (conversation_id, role, content, token_count, timestamp, position) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (conv_id, m.role.value, m.content, m.token_count,
                     m.timestamp.isoformat(), i)
                    for i, m in enumerate(messages)
                ],
            )

    def load_conversation(self, conv_id: str) -> tuple[str, list[Message]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if row is None:
                return None
            name = row["name"]
            msg_rows = conn.execute(
                "SELECT role, content, token_count, timestamp FROM messages "
                "WHERE conversation_id = ? ORDER BY position",
                (conv_id,),
            ).fetchall()
            messages = [
                Message(
                    role=Role(r["role"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in msg_rows
            ]
        return name, messages

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.id, c.name, c.updated_at, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def export_json(self, conv_id: str) -> str:
        result = self.load_conversation(conv_id)
        if result is None:
            raise ValueError(f"Conversation '{conv_id}' not found.")
        name, messages = result
        data = {
            "id": conv_id,
            "name": name,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "token_count": m.token_count,
                }
                for m in messages
            ],
        }
        return json.dumps(data, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add matgpt/storage.py tests/test_storage.py
git commit -m "feat: add SQLite storage with save, load, list, delete, export"
```

---

## Task 8: Session Manager

**Files:**
- Create: `matgpt/session.py`
- Modify: `matgpt/__init__.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `Conversation` from `matgpt.conversation`; `Storage` from `matgpt.storage`; `MatGPTClient` from `matgpt.client`; `ModelManager` from `matgpt.models`; `Config` from `matgpt.config`
- Produces:
  - `SessionManager(config: Config)` — wires up client, model_manager, storage
  - `SessionManager.new_conversation(name: str, system_prompt: str = "") -> Conversation`
  - `SessionManager.load_conversation(conv_id: str) -> Conversation | None`
  - `SessionManager.save_conversation(conv: Conversation) -> None`
  - `SessionManager.list_conversations() -> list[dict]`
  - `SessionManager.delete_conversation(conv_id: str) -> None`
  - `SessionManager.model_manager -> ModelManager` (property)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session.py
from unittest.mock import MagicMock, patch
from matgpt.session import SessionManager
from matgpt.message import Role


def make_session(tmp_path):
    with patch("matgpt.session.MatGPTClient") as MockClient, \
         patch("matgpt.session.ModelManager") as MockMgr:

        mock_client_instance = MagicMock()
        mock_client_instance.list_models.return_value = ["llama-3"]
        mock_client_instance.complete.return_value = "Hello!"
        MockClient.return_value = mock_client_instance

        mock_mgr_instance = MagicMock()
        mock_mgr_instance.current.return_value = "llama-3"
        MockMgr.return_value = mock_mgr_instance

        from matgpt.config import Config
        config = Config(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            db_path=str(tmp_path / "test.db"),
            default_context_window=4096,
        )
        session = SessionManager(config)
        return session, mock_client_instance, mock_mgr_instance


def test_new_conversation_returns_conversation(tmp_path):
    session, _, _ = make_session(tmp_path)
    conv = session.new_conversation("My Chat", system_prompt="Be helpful.")
    assert conv.name == "My Chat"
    history = conv.history()
    assert history[0].role == Role.SYSTEM


def test_save_and_load_conversation(tmp_path):
    session, client, _ = make_session(tmp_path)
    conv = session.new_conversation("Chat 1")
    conv.chat("Hello")
    session.save_conversation(conv)

    loaded = session.load_conversation(conv.id)
    assert loaded is not None
    assert loaded.name == "Chat 1"


def test_list_conversations(tmp_path):
    session, client, _ = make_session(tmp_path)
    c1 = session.new_conversation("Chat A")
    c2 = session.new_conversation("Chat B")
    session.save_conversation(c1)
    session.save_conversation(c2)
    listing = session.list_conversations()
    assert len(listing) == 2


def test_delete_conversation(tmp_path):
    session, client, _ = make_session(tmp_path)
    conv = session.new_conversation("Temp")
    session.save_conversation(conv)
    session.delete_conversation(conv.id)
    assert session.load_conversation(conv.id) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_session.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `matgpt/session.py`**

```python
from __future__ import annotations
import uuid

from matgpt.client import MatGPTClient
from matgpt.config import Config
from matgpt.conversation import Conversation
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
        # Save non-system messages only (system prompt is stored on the object)
        messages = [m for m in conv.history() if True]  # save all including system
        self._storage.save_conversation(conv.id, conv.name, conv.history())

    def load_conversation(self, conv_id: str) -> Conversation | None:
        result = self._storage.load_conversation(conv_id)
        if result is None:
            return None
        name, messages = result

        from matgpt.message import Role
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
```

- [ ] **Step 4: Update `matgpt/__init__.py`**

```python
from matgpt.config import Config, get_config
from matgpt.session import SessionManager
from matgpt.message import Message, Role
from matgpt.conversation import Conversation

__all__ = ["Config", "get_config", "SessionManager", "Message", "Role", "Conversation"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_session.py -v
```
Expected: 4 PASSED

- [ ] **Step 6: Full test suite**

```bash
pytest -v
```
Expected: all PASSED

- [ ] **Step 7: Commit**

```bash
git add matgpt/session.py matgpt/__init__.py tests/test_session.py
git commit -m "feat: add SessionManager — wires all components together"
```

---

## Task 9: README + Integration Smoke Test

**Files:**
- Create: `README.md`
- Create: `examples/quickstart.py`

**Interfaces:**
- Consumes: `SessionManager`, `get_config` from `matgpt`

- [ ] **Step 1: Create `examples/quickstart.py`**

```python
"""
MatGPT quickstart — requires LM Studio running on localhost:1234 with a model loaded.
Run: python examples/quickstart.py
"""
from matgpt import SessionManager, get_config

config = get_config()
session = SessionManager(config)

# Pick a model
models = session.model_manager.available()
if not models:
    print("No models loaded in LM Studio. Load one first.")
    exit(1)

session.model_manager.select(models[0])
print(f"Using model: {models[0]}\n")

# Start a conversation
conv = session.new_conversation(
    name="Quickstart Chat",
    system_prompt="You are a helpful, concise assistant.",
)

print("MatGPT ready. Type 'quit' to exit, 'save' to persist, 'usage' for token info.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "save":
        session.save_conversation(conv)
        print(f"[Saved conversation: {conv.id}]")
        continue
    if user_input.lower() == "usage":
        usage = conv.token_usage()
        print(f"[Tokens — system: {usage['system']}, messages: {usage['messages']}, total: {usage['total']}]")
        continue

    print("MatGPT: ", end="", flush=True)
    for chunk in conv.chat(user_input, stream=True):
        print(chunk, end="", flush=True)
    print()
```

- [ ] **Step 2: Create `README.md`**

```markdown
# MatGPT

Local LLM chat backend powered by LM Studio.

## Setup

```bash
pip install -r requirements.txt
```

Requires LM Studio running with at least one model loaded (default: `http://localhost:1234`).

## Quickstart

```bash
python examples/quickstart.py
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MATGPT_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `MATGPT_DB_PATH` | `~/.matgpt/matgpt.db` | SQLite database path |
| `MATGPT_CONTEXT_WINDOW` | `4096` | Default max token budget |

## Run tests

```bash
pytest -v
```
```

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest -v
```
Expected: all PASSED

- [ ] **Step 4: Final commit**

```bash
git add README.md examples/
git commit -m "docs: add README and quickstart example"
```

---

## Self-Review

**Spec coverage:**
- ✅ Multi-turn conversation with memory — `Conversation`
- ✅ Streaming — `MatGPTClient.stream`, `Conversation.chat(stream=True)`
- ✅ Model selection/switching — `ModelManager`
- ✅ System prompt — `Conversation.set_system_prompt`
- ✅ Context window management / truncation — `context.fit_to_window`
- ✅ Token counting — `count_tokens`, `annotate_tokens`
- ✅ Multiple sessions — `SessionManager.new_conversation`
- ✅ Persistence (save/load/list/delete) — `Storage`, `SessionManager`
- ✅ Export — `Storage.export_json`
- ✅ Configuration via env vars — `Config`, `get_config`

**No placeholders found.**

**Type consistency verified:** All method signatures match across tasks (e.g., `Conversation.chat` returns `str | Iterator[str]`; `fit_to_window` takes `list[Message]` everywhere).
