"""
MatGPT FastAPI server — bridges the frontend to the local LLM backend.

Run:
    pip install fastapi uvicorn
    python -m matgpt.server

Then open frontend/index.html in your browser.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from matgpt.config import get_config
from matgpt.launcher import ensure_lmstudio_running
from matgpt.session import SessionManager

# ── Boot ─────────────────────────────────────────────────────────────────────

ensure_lmstudio_running(verbose=True)
config  = get_config()
session = SessionManager(config)

app = FastAPI(title="MatGPT", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class SelectModelRequest(BaseModel):
    model: str

class NewConversationRequest(BaseModel):
    name: str = "New chat"
    system_prompt: str = "You are a helpful, concise assistant."

class ChatRequest(BaseModel):
    message: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/models")
def list_models() -> dict:
    return {"models": session.model_manager.available()}


@app.post("/models/select")
def select_model(req: SelectModelRequest) -> dict:
    try:
        session.model_manager.select(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"selected": req.model}


@app.get("/conversations")
def list_conversations() -> dict:
    return {"conversations": session.list_conversations()}


@app.post("/conversations")
def new_conversation(req: NewConversationRequest) -> dict:
    conv = session.new_conversation(req.name, system_prompt=req.system_prompt)
    session.save_conversation(conv)
    return {"id": conv.id, "name": conv.name}


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str) -> dict:
    conv = session.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "name": conv.name,
        "messages": [
            {
                "role": m.role.value,
                "content": m.content,
                "token_count": m.token_count,
            }
            for m in conv.history()
        ],
    }


@app.post("/conversations/{conv_id}/chat")
def chat(conv_id: str, req: ChatRequest) -> dict:
    conv = session.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    response = conv.chat(req.message, stream=False)
    session.save_conversation(conv)

    usage = conv.token_usage()
    return {
        "response": response,
        "tokens": usage,
    }


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str) -> dict:
    session.delete_conversation(conv_id)
    return {"deleted": conv_id}


@app.get("/conversations/{conv_id}/export")
def export_conversation(conv_id: str) -> dict:
    try:
        import json
        return json.loads(session.export_conversation(conv_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matgpt.server:app", host="0.0.0.0", port=8000, reload=False)
