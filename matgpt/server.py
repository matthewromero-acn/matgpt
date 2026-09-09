"""
MatGPT FastAPI server — bridges the Next.js frontend to the local LLM backend.

Run:
    python -m matgpt.server
    # or via start.sh from the repo root
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from matgpt.config import get_config
from matgpt.launcher import ensure_lmstudio_running
from matgpt.session import SessionManager

# ── Boot ──────────────────────────────────────────────────────────────────────

ensure_lmstudio_running(verbose=True)

config  = get_config()
session = SessionManager(config)

app = FastAPI(title="MatGPT", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class SelectModelRequest(BaseModel):
    model: str

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

# ── Conversations ─────────────────────────────────────────────────────────────

class NewConversationRequest(BaseModel):
    name: str = "New chat"
    system_prompt: str = "You are a helpful, concise assistant."

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

@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str) -> dict:
    session.delete_conversation(conv_id)
    return {"deleted": conv_id}

# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

@app.post("/conversations/{conv_id}/chat")
def chat(conv_id: str, req: ChatRequest) -> dict:
    conv = session.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    response = conv.chat(req.message, stream=False)
    session.save_conversation(conv)

    return {
        "response": response,
        "tokens": conv.token_usage(),
    }

@app.post("/conversations/{conv_id}/chat/stream")
def chat_stream(conv_id: str, req: ChatRequest):
    conv = session.load_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    def generate():
        try:
            for chunk in conv.chat(req.message, stream=True):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            # Stream exhausted — assistant message has been appended to conv
            session.save_conversation(conv)
            yield f"data: {json.dumps({'done': True, 'tokens': conv.token_usage()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matgpt.server:app", host="127.0.0.1", port=8000, reload=False)
