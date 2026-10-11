"""OpenAI-compatible Chat Completions transport, without provider SDK coupling."""
from dataclasses import dataclass, field
from pathlib import Path
import os
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
import httpx


class APIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    key: str = field(repr=False)
    base_url: str
    model: str

    @classmethod
    def load(cls, path: Path):
        load_dotenv(path, override=False)
        values = [os.environ.get(name, "").strip() for name in
                  ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")]
        if not all(values):
            raise ValueError("Set OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL in .env")
        parsed = urlparse(values[1])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("OPENAI_BASE_URL must be an HTTP(S) base URL without query or fragment")
        if parsed.username or parsed.password:
            raise ValueError("Credentials belong in OPENAI_API_KEY, not the URL")
        return cls(*values)


class ChatClient:
    def __init__(self, settings: Settings, max_tokens: int = 2048, timeout: float = 90,
                 temperature: float = 0):
        self.settings = settings
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.http_requests = 0
        self.http = httpx.Client(timeout=timeout, follow_redirects=False)

    def close(self):
        self.http.close()

    def complete(self, messages: list, tools: list) -> dict:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload = dict(model=self.settings.model, messages=messages, tools=tools,
                       tool_choice="auto", temperature=self.temperature, max_tokens=self.max_tokens)
        for attempt in range(3):
            try:
                self.http_requests += 1
                response = self.http.post(url, json=payload,
                    headers={"Authorization": f"Bearer {self.settings.key}"})
            except httpx.TransportError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise APIError("API transport failure after 3 attempts (URL and credentials omitted)") from None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            if response.is_error:
                # Do not write arbitrary provider error bodies, which can echo secrets.
                raise APIError(f"API HTTP {response.status_code}; check model, endpoint and tool support")
            if response.is_redirect:
                raise APIError("API redirect refused; configure the final API base URL")
            try:
                data = response.json()
                message = data["choices"][0]["message"]
                if not isinstance(message, dict) or not isinstance(data.get("usage", {}), dict):
                    raise ValueError
                return data
            except (ValueError, KeyError, IndexError, TypeError):
                raise APIError("Invalid Chat Completions response") from None
        raise APIError("API attempts exhausted")
