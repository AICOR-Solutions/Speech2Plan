from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.settings import Prompts, default_settings

# %% reply shape


class ReplyKey(StrEnum):
    """Keys the model is asked to produce."""

    CHOICE = "choice"


# %% interpreter


@dataclass
class AnswerInterpreter:
    """Maps what a person said onto one of the answers a question offered."""

    model: LanguageModel = field(kw_only=True)
    prompts: Prompts = field(
        kw_only=True, default_factory=lambda: default_settings().prompts
    )

    def choose(self, question: str, offered: List[str], spoken: str) -> Optional[str]:
        """The offered answer the words mean, or None when none of them does."""
        if not offered:
            return None
        reply = self.model.complete_json(
            CompletionRequest(
                system=self.prompts.answer_to(question, offered),
                user=spoken,
                json_schema=self.reply_schema(offered),
            )
        )
        return reply.get(ReplyKey.CHOICE)

    def reply_schema(self, offered: List[str]) -> dict:
        """The shape of a reply, with the offered answers built into the enum."""
        return {
            "type": "object",
            "properties": {
                str(ReplyKey.CHOICE): {
                    "type": ["string", "null"],
                    "enum": offered + [None],
                }
            },
            "required": [str(ReplyKey.CHOICE)],
            "additionalProperties": False,
        }
