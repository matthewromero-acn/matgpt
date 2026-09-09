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


def test_chat_calls_client_complete():
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
