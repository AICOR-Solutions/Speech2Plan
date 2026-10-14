from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from speech2plan.parsing.backend import CompletionRequest, LanguageModel

# %% a model that answers from a script


@dataclass
class FakeModel(LanguageModel):
    """Returns one canned reply and keeps what it was asked."""

    reply: str = field(kw_only=True)
    requests: List[CompletionRequest] = field(kw_only=True, default_factory=list)

    def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        return self.reply
