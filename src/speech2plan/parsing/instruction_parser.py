from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from hri_actions_msgs.msg import Intent

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    RoleBinding,
    ThematicRole,
)
from speech2plan.exceptions import SymbolNotInVocabulary
from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.vocabulary import Vocabulary

# %% reply shape


class ReplyKey(StrEnum):
    """Keys the model is asked to produce."""

    INTENT = "intent"
    ROLES = "Instituteroles"
    UNFILLED = "unfilled_roles"
    CONFIDENCE = "confidence"
    ENTITY_TYPE = "entity_type"
    QUALIFIERS = "qualifiers"
    SPOKEN_FORM = "spoken_form"
    ATTRIBUTE = "attribute"
    VALUE = "value"


# %% parser


@dataclass
class InstructionParser:
    """Turns an utterance into a ParsedInstruction.

    The model chooses from the vocabulary and nothing else. Every entity type
    it returns is checked against the world, so a hallucinated room type fails
    here instead of reaching the planner.
    """

    model: LanguageModel = field(kw_only=True)
    vocabulary: Vocabulary = field(kw_only=True)

    def parse(
        self, utterance: str, speaker: str = Intent.UNKNOWN_AGENT
    ) -> ParsedInstruction:
        request = CompletionRequest(system=self.system_prompt(), user=utterance)
        reply = json.loads(self.strip_fences(self.model.complete(request)))
        return self.build_instruction(utterance, speaker, reply)

    def system_prompt(self) -> str:
        role_lines = "\n".join(f"- {role}" for role in ThematicRole)
        return (
            "You convert a spoken instruction into one JSON object describing a "
            "ROS4HRI intent.\n\n"
            "Rules:\n"
            "- Use only the intents and entity types listed below.\n"
            "- Name an entity TYPE, never a specific room. For 'the kitchen' "
            "answer entity_type 'Kitchen', and add a floor qualifier only if a "
            "floor was spoken.\n"
            "- A role the speaker left open goes in 'unfilled_roles'. Never "
            "guess. 'this' and 'that' are unfilled unless named.\n"
            "- Answer with JSON only, no prose and no code fences.\n\n"
            f"{self.vocabulary.render()}\n\n"
            f"Thematic roles:\n{role_lines}\n\n"
            "Shape:\n"
            '{"intent": "__intent_bring_object__", '
            '"roles": {"goal": {"entity_type": "Kitchen", "qualifiers": '
            '[{"attribute": "floor", "value": "GroundFloor"}], '
            '"spoken_form": "the kitchen downstairs"}}, '
            '"unfilled_roles": ["object"], "confidence": 0.9}'
        )

    def strip_fences(self, reply: str) -> str:
        stripped = reply.strip()
        if not stripped.startswith("```"):
            return stripped
        without_opening = stripped.split("\n", 1)[1]
        return without_opening.rsplit("```", 1)[0]

    def build_instruction(
        self, utterance: str, speaker: str, reply: dict
    ) -> ParsedInstruction:
        return ParsedInstruction(
            utterance=utterance,
            speaker=speaker,
            intent=IntentName(reply[ReplyKey.INTENT]),
            roles=self.build_roles(reply.get(ReplyKey.ROLES, {})),
            unfilled_roles=[
                ThematicRole(name) for name in reply.get(ReplyKey.UNFILLED, [])
            ],
            confidence=float(reply.get(ReplyKey.CONFIDENCE, 0.0)),
        )

    def build_roles(self, payload: dict) -> List[RoleBinding]:
        bindings = []
        for role_name, constraint_payload in payload.items():
            constraint = self.build_constraint(constraint_payload)
            if constraint is None:
                continue
            bindings.append(
                RoleBinding(role=ThematicRole(role_name), constraint=constraint)
            )
        return bindings

    def build_constraint(self, payload: Optional[dict]) -> Optional[EntityConstraint]:
        if payload is None:
            return None
        entity_type = payload.get(ReplyKey.ENTITY_TYPE)
        if not entity_type:
            return None
        if entity_type not in self.vocabulary.entity_types:
            raise SymbolNotInVocabulary(
                entity_type=entity_type,
                known_entity_types=self.vocabulary.entity_types,
            )
        return EntityConstraint(
            entity_type=entity_type,
            qualifiers=[
                Qualifier(
                    attribute=item[ReplyKey.ATTRIBUTE], value=item[ReplyKey.VALUE]
                )
                for item in payload.get(ReplyKey.QUALIFIERS, [])
            ],
            spoken_form=payload.get(ReplyKey.SPOKEN_FORM),
        )
