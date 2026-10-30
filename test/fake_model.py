from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from speech2plan.parsing.backend import CompletionRequest, LanguageModel

# %% a model that answers from a script


@dataclass
class FakeModel(LanguageModel):
    """Returns canned replies and keeps what it was asked.

    The scripted `replies` come first, one per request, then `reply` for good.
    """

    reply: str = field(kw_only=True, default="")
    replies: List[str] = field(kw_only=True, default_factory=list)
    requests: List[CompletionRequest] = field(kw_only=True, default_factory=list)

    def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        if self.replies:
            return self.replies.pop(0)
        return self.reply
