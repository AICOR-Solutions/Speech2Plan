from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Tuple

from hri_actions_msgs.msg import Intent
from semantic_digital_twin.world import World

from speech2plan.contract import (
    EntityConstraint,
    GenericType,
    IntentName,
    ParsedInstruction,
    Qualifier,
    Relation,
    RoleBinding,
    RoleConfidence,
    SpatialRelation,
    ThematicRole,
)
from speech2plan.exceptions import UnusableReply
from speech2plan.grounding import Grounder
from speech2plan.parsing.backend import CompletionRequest, LanguageModel, parse_json
from speech2plan.placement import QualifierAttribute
from speech2plan.settings import Prompts, default_settings
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
    RELATIONS = "relations"
    SPOKEN_FORM = "spoken_form"
    ATTRIBUTE = "attribute"
    VALUE = "value"
    RELATION = "relation"
    REFERENCE = "reference"


# %% one exchange


@dataclass
class Exchange:
    """One reply of the model and what came back to it."""

    reply: str = field(kw_only=True)
    response: str = field(kw_only=True)
    """A <result> of a query, or why the reply was <rejected>."""

    answered_query: bool = field(kw_only=True, default=False)


# %% parser


@dataclass
class InstructionParser:
    """Turns an utterance into a ParsedInstruction.

    A reply that breaks the rules is returned to the model with what was wrong,
    because the model named a type or a floor the world lacks, not the speaker.
    """

    model: LanguageModel = field(kw_only=True)
    vocabulary: Vocabulary = field(kw_only=True)
    maximum_attempts: int = field(kw_only=True, default=2)
    """How many replies may break the rules before the utterance is given up."""

    nesting: int = field(kw_only=True, default=2)
    """How deep relations may nest: the cloth in the cabinet in the bathroom."""

    spoken_qualifiers: Tuple[QualifierAttribute, ...] = field(
        kw_only=True, default=(QualifierAttribute.FLOOR,)
    )
    """The qualifiers the model may state. Rooms are references of an `in`
    relation, and names are for the dialogue to pin an answer with."""

    prompts: Prompts = field(
        kw_only=True, default_factory=lambda: default_settings().prompts
    )

    @classmethod
    def vocabulary_of(cls, world: World) -> Vocabulary:
        """What this parser tells the model about a world."""
        return Vocabulary.from_world(world)

    @classmethod
    def for_world(
        cls, world: World, model: LanguageModel, grounder: Optional[Grounder] = None
    ) -> InstructionParser:
        """A parser for one world, which `grounder` grounds against if given."""
        return cls(model=model, vocabulary=cls.vocabulary_of(world))

    def follow(self, world: World) -> None:
        """Read the vocabulary afresh from a world that changed."""
        self.vocabulary = self.vocabulary_of(world)

    # %% the conversation

    def parse(
        self, utterance: str, speaker: str = Intent.UNKNOWN_AGENT
    ) -> ParsedInstruction:
        """Parse one utterance attributed to `speaker`.

        Every reply and what came back to it is sent again with the next request,
        which a stateless backend needs.
        """
        exchanges: List[Exchange] = []
        rejected = 0
        while True:
            reply = self.model.complete(self.request(utterance, exchanges))
            try:
                parsed = parse_json(reply)
            except json.JSONDecodeError as error:
                problems = [f"the reply is no JSON object: {error}"]
            else:
                result = self.query_result(parsed, exchanges)
                if result is not None:
                    exchanges.append(
                        Exchange(
                            reply=reply,
                            response=f"<result>{result}</result>",
                            answered_query=True,
                        )
                    )
                    continue
                problems = self.problems_with(parsed)
                if not problems:
                    return self.build_instruction(utterance, speaker, parsed)
            rejected += 1
            if rejected >= self.maximum_attempts:
                raise UnusableReply(utterance=utterance, problems=problems)
            exchanges.append(
                Exchange(
                    reply=reply, response=f"<rejected>{'; '.join(problems)}</rejected>"
                )
            )

    def query_result(self, reply: object, exchanges: List[Exchange]) -> Optional[str]:
        """The answer to a reply that queries the world, or None when it does not.
        This parser lets the model query nothing."""
        return None

    def request(self, utterance: str, exchanges: List[Exchange]) -> CompletionRequest:
        """The utterance and every exchange about it so far."""
        lines = [f"<utterance>{utterance}</utterance>"]
        for exchange in exchanges:
            lines.extend([exchange.reply, exchange.response])
        return CompletionRequest(
            system=self.system_prompt(),
            user="\n".join(lines),
            json_schema=self.reply_schema(),
        )

    # %% the prompt

    def system_prompt(self) -> str:
        """The instructions, then the world, so the instructions stay a fixed prefix."""
        role_lines = "\n".join(f"- {role}" for role in ThematicRole)
        relation_lines = "\n".join(
            f"- {relation}: {self.prompts.relations[relation]}"
            for relation in self.vocabulary.relations
        )
        return "\n\n".join(
            [
                *self.rules(),
                f"Thematic roles:\n{role_lines}",
                f"Relations:\n{relation_lines}",
                *self.examples(),
                self.world(),
            ]
        )

    def rules(self) -> List[str]:
        return [self.prompts.rules]

    def examples(self) -> List[str]:
        return [self.prompts.examples]

    def world(self) -> str:
        """The world as the prompt states it: every type, and what tells the
        instances of a type apart."""
        return self.vocabulary.render()

    # %% the reply schema

    def reply_schema(self) -> dict:
        """The shape of a reply, with the vocabulary built into the enums."""
        constraint = self.constraint_schema(nesting=self.nesting)
        constraint["type"] = ["object", "null"]
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

    def constraint_schema(self, nesting: int) -> dict:
        """One entity description, whose references relate `nesting - 1` deep."""
        properties = {
            str(ReplyKey.ENTITY_TYPE): {
                "type": ["string", "null"],
                "enum": self.known_types() + [None],
            },
            str(ReplyKey.QUALIFIERS): self.qualifiers_schema(),
            str(ReplyKey.SPOKEN_FORM): {"type": ["string", "null"]},
        }
        if nesting > 0:
            properties[str(ReplyKey.RELATIONS)] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        str(ReplyKey.RELATION): {
                            "type": "string",
                            "enum": [str(r) for r in self.vocabulary.relations],
                        },
                        str(ReplyKey.REFERENCE): self.constraint_schema(
                            nesting=nesting - 1
                        ),
                    },
                    "required": [str(ReplyKey.RELATION), str(ReplyKey.REFERENCE)],
                    "additionalProperties": False,
                },
            }
        return {
            "type": "object",
            "properties": properties,
            "required": [str(ReplyKey.ENTITY_TYPE)],
            "additionalProperties": False,
        }

    def known_types(self) -> List[str]:
        """The types of the world, and the one for a thing left unnamed."""
        return self.vocabulary.entity_types + [str(GenericType.THING)]

    def qualifiers_schema(self) -> dict:
        """The qualifiers, which may only name a floor the world has."""
        if not self.vocabulary.floors:
            return {"type": "array", "maxItems": 0}
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    str(ReplyKey.ATTRIBUTE): {
                        "type": "string",
                        "enum": [
                            str(attribute) for attribute in self.spoken_qualifiers
                        ],
                    },
                    str(ReplyKey.VALUE): {
                        "type": "string",
                        "enum": self.vocabulary.floors,
                    },
                },
                "required": [str(ReplyKey.ATTRIBUTE), str(ReplyKey.VALUE)],
                "additionalProperties": False,
            },
        }

    # %% checking a reply, since not every backend constrains decoding

    def problems_with(self, reply: object) -> List[str]:
        """Everything in the reply that breaks the schema, in words for the model."""
        if not isinstance(reply, dict):
            return ["the reply must be one JSON object"]
        problems = []
        intents = [str(intent) for intent in self.vocabulary.intents]
        if reply.get(ReplyKey.INTENT) not in intents:
            problems.append(
                f"intent {reply.get(ReplyKey.INTENT)!r} is not one of {intents}"
            )
        roles = reply.get(ReplyKey.ROLES, {})
        if not isinstance(roles, dict):
            return problems + ["roles must be an object"]
        known_roles = [str(role) for role in ThematicRole]
        for name, payload in roles.items():
            if name not in known_roles:
                problems.append(f"role {name!r} is not one of {known_roles}")
            elif payload is not None:
                problems.extend(self.constraint_problems(f"roles.{name}", payload))
        unfilled = reply.get(ReplyKey.UNFILLED, [])
        if not isinstance(unfilled, list):
            problems.append("unfilled_roles must be a list of roles")
            unfilled = []
        for name in unfilled:
            if name not in known_roles:
                problems.append(f"unfilled role {name!r} is not one of {known_roles}")
        confidence = reply.get(ReplyKey.CONFIDENCE, {})
        if (
            not isinstance(confidence, dict)
            or not set(confidence) <= {str(ReplyKey.INTENT), *known_roles}
            or not all(isinstance(score, (int, float)) for score in confidence.values())
        ):
            problems.append(
                "confidence must map 'intent' and the filled roles to numbers"
            )
        return problems

    def constraint_problems(
        self, path: str, payload: object, nesting: Optional[int] = None
    ) -> List[str]:
        """What is wrong with one entity description, whose references may relate
        `nesting` deep, as deep as the parser allows unless given."""
        nesting = self.nesting if nesting is None else nesting
        if not isinstance(payload, dict):
            return [f"{path} must be an object or null"]
        problems = []
        entity_type = payload.get(ReplyKey.ENTITY_TYPE)
        if entity_type is not None and entity_type not in self.known_types():
            problems.append(
                f"{path}.entity_type {entity_type!r} is not in the world. Use one of "
                f"{self.known_types()}, or null with the speaker's words in "
                "spoken_form"
            )
        qualifiers = payload.get(ReplyKey.QUALIFIERS) or []
        relations = payload.get(ReplyKey.RELATIONS) or []
        if not isinstance(qualifiers, list) or not isinstance(relations, list):
            return problems + [f"{path} needs lists of qualifiers and relations"]
        for qualifier in qualifiers:
            if not isinstance(qualifier, dict):
                problems.append(f"{path} has a qualifier that is no object")
                continue
            attribute = qualifier.get(ReplyKey.ATTRIBUTE)
            value = qualifier.get(ReplyKey.VALUE)
            if attribute not in self.spoken_qualifiers:
                problems.append(
                    f"{path} has qualifier {attribute!r}; only "
                    f"{[str(a) for a in self.spoken_qualifiers]} exist, and a room is "
                    "the reference of an 'in' relation"
                )
            elif value not in self.vocabulary.floors:
                problems.append(
                    f"{path} names floor {value!r}; the floors are "
                    f"{self.vocabulary.floors}"
                )
        if relations and nesting <= 0:
            problems.append(
                f"{path} relates too deep; relations nest at most {self.nesting} deep"
            )
        relation_names = [str(relation) for relation in self.vocabulary.relations]
        for index, relation in enumerate(relations if nesting > 0 else []):
            if not isinstance(relation, dict):
                problems.append(f"{path}.relations[{index}] is no object")
                continue
            if relation.get(ReplyKey.RELATION) not in relation_names:
                problems.append(
                    f"{path}.relations[{index}] is {relation.get(ReplyKey.RELATION)!r}, "
                    f"not one of {relation_names}"
                )
            problems.extend(
                self.constraint_problems(
                    f"{path}.relations[{index}].reference",
                    relation.get(ReplyKey.REFERENCE),
                    nesting=nesting - 1,
                )
            )
        return problems

    # %% building the instruction

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
        """An EntityConstraint, or None when the role was left unfilled.

        A description without a type stands for a kind of thing the world lacks,
        which only the speaker's words describe.
        """
        if payload is None:
            return None
        entity_type = payload.get(ReplyKey.ENTITY_TYPE)
        spoken_form = payload.get(ReplyKey.SPOKEN_FORM)
        if not entity_type and not spoken_form:
            return None
        return EntityConstraint(
            entity_type=entity_type or None,
            qualifiers=[
                Qualifier(
                    attribute=item[ReplyKey.ATTRIBUTE], value=item[ReplyKey.VALUE]
                )
                for item in payload.get(ReplyKey.QUALIFIERS) or []
            ],
            relations=[
                Relation(
                    relation=SpatialRelation(item[ReplyKey.RELATION]),
                    reference=reference,
                )
                for item in payload.get(ReplyKey.RELATIONS) or []
                if (reference := self.build_constraint(item[ReplyKey.REFERENCE]))
                is not None
            ],
            spoken_form=spoken_form,
        )
