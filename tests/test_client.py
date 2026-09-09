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
    mock_openai.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])

    with patch("matgpt.client.OpenAI", return_value=mock_openai):
        client = MatGPTClient(config)
        messages = [Message(role=Role.USER, content="Hi")]
        chunks = list(client.stream(messages, model="test-model"))

    assert chunks == ["Hello", " world"]
    # Verify stream=True was passed
    call_kwargs = mock_openai.chat.completions.create.call_args[1]
    assert call_kwargs.get("stream") is True
