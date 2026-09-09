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
