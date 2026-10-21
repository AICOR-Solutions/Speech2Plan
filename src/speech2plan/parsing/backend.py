from __future__ import annotations

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
