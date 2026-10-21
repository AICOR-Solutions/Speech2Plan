from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import requests

from speech2plan.exceptions import ApiKeyNotFound, LanguageModelCallFailed
from speech2plan.parsing.backend import CompletionRequest, LanguageModel

# %% credentials


def read_api_key(env_file: Path, key_name: str) -> str:
    """The value of `key_name` in a KEY=value file."""
    for line in env_file.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == key_name:
            return value.strip().strip("\"'")
    raise ApiKeyNotFound(key_name=key_name, env_file=str(env_file))


# %% backend


@dataclass
class OpenRouterModel(LanguageModel):
    """Calls any model OpenRouter serves, through its OpenAI shaped API."""

    model: str = field(kw_only=True, default="anthropic/claude-sonnet-4.5")
    endpoint: str = field(
        kw_only=True, default="https://openrouter.ai/api/v1/chat/completions"
    )
    timeout_seconds: float = field(kw_only=True, default=120.0)
    application_name: str = field(kw_only=True, default="Speech2Plan")
    env_file: Path = field(
        kw_only=True,
        default_factory=lambda: Path(__file__).resolve().parents[3] / ".env",
    )
    key_name: str = field(kw_only=True, default="OPENROUTER_API_KEY")

    def complete(self, request: CompletionRequest) -> str:
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {read_api_key(self.env_file, self.key_name)}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": self.application_name,
            },
            json=self.body(request),
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise LanguageModelCallFailed(
                command=f"POST {self.endpoint} ({self.model})",
                exit_code=response.status_code,
                standard_error=response.text,
            )
        return response.json()["choices"][0]["message"]["content"]

    def body(self, request: CompletionRequest) -> dict:
        """The request body, with a schema attached when one is given."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if request.json_schema is None:
            return body
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "parsed_instruction",
                "strict": True,
                "schema": request.json_schema,
            },
        }
        body["provider"] = {"require_parameters": True}
        return body
