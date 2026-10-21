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
    RoleConfidence,
    ThematicRole,
)
from speech2plan.exceptions import SymbolNotInVocabulary
from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.vocabulary import Vocabulary

# %% reply shape


class ReplyKey(StrEnum):
    """Keys the model is asked to produce."""

    INTENT = "intent"
    ROLES = "roles"
    UNFILLED = "unfilled_roles"
    CONFIDENCE = "confidence"
    ENTITY_TYPE = "entity_type"
    QUALIFIERS = "qualifiers"
    SPOKEN_FORM = "spoken_form"
    ATTRIBUTE = "attribute"
    VALUE = "value"


# %% replies


def strip_fences(reply: str) -> str:
    """`reply` without a surrounding markdown code fence."""
    stripped = reply.strip()
    if not stripped.startswith("```"):
        return stripped
    return stripped.split("\n", 1)[1].rsplit("```", 1)[0]


# %% parser


@dataclass
class InstructionParser:
    """Turns an utterance into a ParsedInstruction."""

    model: LanguageModel = field(kw_only=True)
    vocabulary: Vocabulary = field(kw_only=True)

    def parse(
        self, utterance: str, speaker: str = Intent.UNKNOWN_AGENT
    ) -> ParsedInstruction:
        """Parse one utterance attributed to `speaker`."""
        request = CompletionRequest(
            system=self.system_prompt(),
            user=utterance,
            json_schema=self.reply_schema(),
        )
        reply = json.loads(strip_fences(self.model.complete(request)))
        return self.build_instruction(utterance, speaker, reply)

    def system_prompt(self) -> str:
        """The instructions and vocabulary sent with every utterance."""
        role_lines = "\n".join(f"- {role}" for role in ThematicRole)
        return (
            "You convert a spoken instruction into one JSON object describing a "
            "ROS4HRI intent.\n\n"
            "Rules:\n"
            "- Use only the intents and entity types listed below.\n"
            "- Name an entity TYPE, never a specific room. For 'the kitchen' "
            "answer entity_type 'Kitchen', and add a floor qualifier only if a "
            "floor was spoken.\n"
            "- 'goal' is where something goes, a room or a surface. "
            "'recipient' is a person who receives something. 'object' is what "
            "is handled.\n"
            "- A role the speaker left open goes in 'unfilled_roles'. Never "
            "guess. 'this' and 'that' are unfilled unless named.\n"
            "- Score confidence separately for the intent and for each role you "
            "filled, between 0 and 1.\n"
            "- Answer with JSON only, no prose and no code fences.\n\n"
            f"{self.vocabulary.render()}\n\n"
            f"Thematic roles:\n{role_lines}\n\n"
            "Shape:\n"
            '{"intent": "__intent_bring_object__", '
            '"roles": {"goal": {"entity_type": "Kitchen", "qualifiers": '
            '[{"attribute": "floor", "value": "GroundFloor"}], '
            '"spoken_form": "the kitchen downstairs"}}, '
            '"unfilled_roles": ["object"], '
            '"confidence": {"intent": 0.95, "goal": 0.9}}'
        )

    def reply_schema(self) -> dict:
        """The shape of a reply, with the vocabulary built into the enums."""
        constraint = {
            "type": ["object", "null"],
            "properties": {
                str(ReplyKey.ENTITY_TYPE): {
                    "type": ["string", "null"],
                    "enum": self.vocabulary.entity_types + [None],
                },
                str(ReplyKey.QUALIFIERS): {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            str(ReplyKey.ATTRIBUTE): {"type": "string"},
                            str(ReplyKey.VALUE): {"type": "string"},
                        },
                        "required": [
                            str(ReplyKey.ATTRIBUTE),
                            str(ReplyKey.VALUE),
                        ],
                        "additionalProperties": False,
                    },
                },
                str(ReplyKey.SPOKEN_FORM): {"type": ["string", "null"]},
            },
            "required": [str(ReplyKey.ENTITY_TYPE)],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                str(ReplyKey.INTENT): {
                    "type": "string",
                    "enum": [str(intent) for intent in self.vocabulary.intents],
                },
                str(ReplyKey.ROLES): {
                    "type": "object",
                    "properties": {str(role): constraint for role in ThematicRole},
                    "additionalProperties": False,
                },
                str(ReplyKey.UNFILLED): {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [str(role) for role in ThematicRole],
                    },
                },
                str(ReplyKey.CONFIDENCE): {
                    "type": "object",
                    "properties": {
                        str(ReplyKey.INTENT): {"type": "number"},
                        **{str(role): {"type": "number"} for role in ThematicRole},
                    },
                    "additionalProperties": False,
                },
            },
            "required": [str(ReplyKey.INTENT), str(ReplyKey.ROLES)],
            "additionalProperties": False,
        }

    def build_instruction(
        self, utterance: str, speaker: str, reply: dict
    ) -> ParsedInstruction:
        """A ParsedInstruction from the model's JSON reply."""
        return ParsedInstruction(
            utterance=utterance,
            speaker=speaker,
            intent=IntentName(reply[ReplyKey.INTENT]),
            roles=self.build_roles(reply.get(ReplyKey.ROLES, {})),
            unfilled_roles=[
                ThematicRole(name) for name in reply.get(ReplyKey.UNFILLED, [])
            ],
            intent_confidence=float(
                reply.get(ReplyKey.CONFIDENCE, {}).get(ReplyKey.INTENT, 0.0)
            ),
            confidence_per_role=self.build_confidences(
                reply.get(ReplyKey.CONFIDENCE, {})
            ),
        )

    def build_confidences(self, payload: dict) -> List[RoleConfidence]:
        """A score for every role the model scored."""
        return [
            RoleConfidence(role=ThematicRole(name), confidence=float(score))
            for name, score in payload.items()
            if name != ReplyKey.INTENT
        ]

    def build_roles(self, payload: dict) -> List[RoleBinding]:
        """A binding for every role the model filled."""
        bindings = (
            (role_name, self.build_constraint(constraint_payload))
            for role_name, constraint_payload in payload.items()
        )
        return [
            RoleBinding(role=ThematicRole(role_name), constraint=constraint)
            for role_name, constraint in bindings
            if constraint is not None
        ]

    def build_constraint(self, payload: Optional[dict]) -> Optional[EntityConstraint]:
        """An EntityConstraint, or None when the role was left unfilled."""
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
