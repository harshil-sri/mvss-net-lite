import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def call_llm(prompt: str, system: str = None) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        preview = prompt[:120] + "..." if len(prompt) > 120 else prompt
        return f"[stub response] Please set OPENROUTER_API_KEY to use the chat. Preview: {preview}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "DocForge",
        "Content-Type": "application/json"
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "openrouter/free",
        "messages": messages
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff
                continue
            return f"Error calling OpenRouter: {str(e)}"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"Error calling OpenRouter: {str(e)}"

    return "Error calling OpenRouter: Max retries exceeded"
