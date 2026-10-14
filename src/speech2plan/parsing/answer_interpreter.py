from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.parsing.instruction_parser import strip_fences

# %% reply shape


class ReplyKey(StrEnum):
    """Keys the model is asked to produce."""

    CHOICE = "choice"


# %% interpreter


@dataclass
class AnswerInterpreter:
    """Maps what a person said onto one of the answers a question offered."""

    model: LanguageModel = field(kw_only=True)

    def choose(self, question: str, offered: List[str], spoken: str) -> Optional[str]:
        """The offered answer the words mean, or None when none of them does."""
        if not offered:
            return None
        reply = json.loads(
            strip_fences(
                self.model.complete(
                    CompletionRequest(
                        system=self.system_prompt(question, offered),
                        user=spoken,
                        json_schema=self.reply_schema(offered),
                    )
                )
            )
        )
        return reply.get(ReplyKey.CHOICE)

    def system_prompt(self, question: str, offered: List[str]) -> str:
        """The instructions and the options sent with every answer."""
        option_lines = "\n".join(f"- {option}" for option in offered)
        return (
            "A robot asked a person a question and the person answered. Decide "
            "which of the offered answers the person meant.\n\n"
            f"Question:\n{question}\n\n"
            f"Offered answers:\n{option_lines}\n\n"
            "Rules:\n"
            "- Answer with one of the offered answers, copied exactly.\n"
            "- Answer null when the person meant none of them, or when two of "
            "them fit equally well.\n"
            "- Spelling mistakes, shorthand and missing words are expected. "
            "'ground flor' means the answer about the ground floor.\n"
            "- Answer with JSON only, no prose and no code fences.\n\n"
            'Shape:\n{"choice": "<one of the offered answers, or null>"}'
        )

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
