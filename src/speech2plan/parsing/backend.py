from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# %% request


@dataclass
class CompletionRequest:
    """One call to a language model."""

    system: str = field(kw_only=True)
    user: str = field(kw_only=True)
    json_schema: Optional[dict] = field(kw_only=True, default=None)
    """The shape the reply must take. Backends that cannot constrain decoding ignore it."""


# %% backend interface


class LanguageModel(ABC):
    """A source of completions."""

    @abstractmethod
    def complete(self, request: CompletionRequest) -> str:
        """Return the model's reply as text."""

    def complete_json(self, request: CompletionRequest) -> dict:
        """The model's reply parsed as JSON, a surrounding code fence ignored."""
        return parse_json(self.complete(request))


def parse_json(reply: str) -> dict:
    """A reply parsed as JSON, a surrounding code fence ignored."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = reply.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(reply)
