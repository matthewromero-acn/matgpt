"""
Quick test script for a locally running LM Studio server.
Automatically starts LM Studio if it isn't already running.
"""

from matgpt.launcher import ensure_lmstudio_running
ensure_lmstudio_running()  # no-op if already up

from openai import OpenAI

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"  # LM Studio ignores this, but openai SDK requires it


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def list_models(client: OpenAI) -> list[str]:
    models = client.models.list()
    return [m.id for m in models.data]


def chat(client: OpenAI, model: str, user_message: str, stream: bool = True) -> str:
    """Send a chat message; streams output to stdout if stream=True."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    if stream:
        full_response = ""
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=0.7,
        ) as s:
            for text in s.text_stream:
                print(text, end="", flush=True)
                full_response += text
        print()  # newline after stream ends
        return full_response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        result = response.choices[0].message.content
        print(result)
        return result


def main():
    client = get_client()

    # 1. List available models
    models = list_models(client)
    if not models:
        print("No models loaded in LM Studio. Load one first.")
        return

    print("=== Loaded models ===")
    for m in models:
        print(f"  {m}")
    print()

    # 2. Use the first loaded model
    model = models[0]
    print(f"=== Chatting with: {model} ===\n")

    chat(client, model, "What is 2 + 2? Explain your reasoning briefly.")


if __name__ == "__main__":
    main()
