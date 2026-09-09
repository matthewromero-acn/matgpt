"""
MatGPT quickstart — requires LM Studio running on localhost:1234 with a model loaded.
Run: python examples/quickstart.py
"""
from matgpt import SessionManager, get_config
from matgpt.launcher import ensure_lmstudio_running

ensure_lmstudio_running()

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
