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
