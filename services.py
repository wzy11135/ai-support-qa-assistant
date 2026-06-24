import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - keeps local fallback usable before dependency install.
    requests = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None


load_dotenv()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    used_provider: str


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> LLMResponse:
        if not self.enabled or requests is None:
            return LLMResponse(content="", used_provider="local-fallback")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=45)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, used_provider=self.model)
