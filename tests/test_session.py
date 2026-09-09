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


def test_export_conversation(tmp_path):
    session, client, _ = make_session(tmp_path)
    conv = session.new_conversation("Export Test")
    conv.chat("Hello")
    session.save_conversation(conv)
    json_str = session.export_conversation(conv.id)
    import json
    data = json.loads(json_str)
    assert data["id"] == conv.id
    assert data["name"] == "Export Test"
    assert len(data["messages"]) >= 1
