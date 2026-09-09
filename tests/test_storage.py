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
