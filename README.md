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
