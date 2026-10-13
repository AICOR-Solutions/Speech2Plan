from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# %% request


@dataclass
class CompletionRequest:
    """One call to a language model.

    Images are present for deictic reference ("bring this"), which needs a
    model that can see. A backend that cannot take images raises instead of
    dropping them.
    """

    system: str = field(kw_only=True)
    user: str = field(kw_only=True)
    images: List[Path] = field(kw_only=True, default_factory=list)


# %% backend interface


class LanguageModel(ABC):
    """A source of completions.

    Implementations so far: the Claude command line. An HTTP backend and a
    local model both fit here without the parser changing.
    """

    @abstractmethod
    def complete(self, request: CompletionRequest) -> str:
        """Return the model's reply as text."""
