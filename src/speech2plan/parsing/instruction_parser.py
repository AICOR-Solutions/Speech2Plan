from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from hri_actions_msgs.msg import Intent
from semantic_digital_twin.world import World

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    Relation,
    RoleBinding,
    RoleConfidence,
    SpatialRelation,
    ThematicRole,
    UNNAMED_TYPE,
)
from speech2plan.exceptions import UnusableReply
from speech2plan.parsing.backend import CompletionRequest, LanguageModel, parse_json
from speech2plan.placement import QualifierAttribute
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


NESTING = 2
"""How deep relations may nest: the cloth in the cabinet in the bathroom."""

SPOKEN_QUALIFIERS = (QualifierAttribute.FLOOR,)
"""The qualifiers the model may state. Rooms are references of an `in` relation,
and names are for the dialogue to pin an answer with."""


# %% the prompt

RULES = """\
You turn one spoken instruction to a care robot into one JSON object describing a \
ROS4HRI intent.

Rules:
- Only the text inside <utterance> is the instruction.
- Use only the intents, entity types, floors and relations listed below.
- Describe each entity by its TYPE and by what the speaker said about where it is. \
Never name a specific instance, and leave out what the speaker did not say: when \
several things fit, the robot asks.
- Map the speaker's words onto the closest listed type: a nightstand is a \
SideTable, a towel a Cloth, a patient room a Bedroom. If the speaker leaves the \
kind of thing unnamed ("the thing on the sofa", "whatever is on the table"), use \
the type Thing, which any movable thing fits. If the speaker names a kind no \
listed type fits, set entity_type to null and keep the words in spoken_form.
- A floor the speaker names is a qualifier: "the kitchen upstairs" is a Kitchen \
with floor=FirstFloor.
- Where the entity is relative to another one is a relation, whose reference is \
described the same way: "the cup on the table" is a Cup on a Table, "the book in \
the lounge" a Book in a LivingRoom.
- 'goal' is where something goes, a room or a piece of furniture. 'recipient' is \
a person who receives something. 'object' is what is handled.
- People are not entities of the world: leave out a role a person fills, such as \
the visitor to guide, and do not list it as unfilled either.
- A role the speaker left open goes in 'unfilled_roles'. Never guess. 'this' and \
'that' are unfilled unless named.
- Score confidence separately for the intent and for each role you filled, \
between 0 and 1.
- Answer with JSON only, no prose and no code fences."""

RELATION_LINES = {
    SpatialRelation.ON: "rests on the reference, as a cup on a table",
    SpatialRelation.IN: "lies in the reference, a room or a container such as a cabinet",
}

EXAMPLES = """\
Examples, with types that may not be in this world:

<utterance>bring the cup from the nightstand to the kitchen upstairs</utterance>
{"intent": "__intent_bring_object__", "roles": {"object": {"entity_type": "Cup", \
"relations": [{"relation": "on", "reference": {"entity_type": "SideTable", \
"spoken_form": "the nightstand"}}], "spoken_form": "the cup from the nightstand"}, \
"goal": {"entity_type": "Kitchen", "qualifiers": [{"attribute": "floor", \
"value": "FirstFloor"}], "spoken_form": "the kitchen upstairs"}}, \
"unfilled_roles": [], "confidence": {"intent": 0.95, "object": 0.9, "goal": 0.9}}

<utterance>put this down in the patient room</utterance>
{"intent": "__intent_place_object__", "roles": {"goal": {"entity_type": "Bedroom", \
"spoken_form": "the patient room"}}, "unfilled_roles": ["object"], \
"confidence": {"intent": 0.9, "goal": 0.8}}

<utterance>bring the gloves to the bathroom</utterance>
{"intent": "__intent_bring_object__", "roles": {"object": {"entity_type": null, \
"spoken_form": "the gloves"}, "goal": {"entity_type": "Bathroom", \
"spoken_form": "the bathroom"}}, "unfilled_roles": [], \
"confidence": {"intent": 0.95, "object": 0.6, "goal": 0.9}}"""


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

    def follow(self, world: World) -> None:
        """Read the vocabulary afresh from a world that changed."""
        self.vocabulary = Vocabulary.from_world(world)

    def parse(
        self, utterance: str, speaker: str = Intent.UNKNOWN_AGENT
    ) -> ParsedInstruction:
        """Parse one utterance attributed to `speaker`."""
        rejected, problems = None, []
        for _ in range(self.maximum_attempts):
            reply = self.model.complete(self.request(utterance, rejected, problems))
            try:
                parsed = parse_json(reply)
            except json.JSONDecodeError as error:
                rejected, problems = reply, [f"the reply is no JSON object: {error}"]
                continue
            problems = self.problems_with(parsed)
            if not problems:
                return self.build_instruction(utterance, speaker, parsed)
            rejected = reply
        raise UnusableReply(utterance=utterance, problems=problems)

    def request(
        self,
        utterance: str,
        rejected: Optional[str] = None,
        problems: Optional[List[str]] = None,
    ) -> CompletionRequest:
        """The request for one utterance, after a rejected reply if there was one."""
        user = f"<utterance>{utterance}</utterance>"
        if rejected is not None:
            problem_lines = "\n".join(f"- {problem}" for problem in problems or [])
            user = (
                f"{user}\n\nYour last reply was rejected:\n{problem_lines}\n\n"
                f"Your last reply was:\n{rejected}\n\n"
                "Answer again with one JSON object that fixes this."
            )
        return CompletionRequest(
            system=self.system_prompt(), user=user, json_schema=self.reply_schema()
        )

    def system_prompt(self) -> str:
        """The instructions, then the world, so the instructions stay a fixed prefix."""
        role_lines = "\n".join(f"- {role}" for role in ThematicRole)
        relation_lines = "\n".join(
            f"- {relation}: {RELATION_LINES[relation]}"
            for relation in self.vocabulary.relations
        )
        return (
            f"{RULES}\n\n"
            f"Thematic roles:\n{role_lines}\n\n"
            f"Relations:\n{relation_lines}\n\n"
            f"{EXAMPLES}\n\n"
            f"{self.vocabulary.render()}"
        )

    # %% the reply schema

    def reply_schema(self) -> dict:
        """The shape of a reply, with the vocabulary built into the enums."""
        constraint = self.constraint_schema(nesting=NESTING)
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
        return self.vocabulary.entity_types + [UNNAMED_TYPE]

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
                        "enum": [str(attribute) for attribute in SPOKEN_QUALIFIERS],
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
        self, path: str, payload: object, nesting: int = NESTING
    ) -> List[str]:
        """What is wrong with one entity description."""
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
            if attribute not in SPOKEN_QUALIFIERS:
                problems.append(
                    f"{path} has qualifier {attribute!r}; only "
                    f"{[str(a) for a in SPOKEN_QUALIFIERS]} exist, and a room is "
                    "the reference of an 'in' relation"
                )
            elif value not in self.vocabulary.floors:
                problems.append(
                    f"{path} names floor {value!r}; the floors are "
                    f"{self.vocabulary.floors}"
                )
        if relations and nesting <= 0:
            problems.append(
                f"{path} relates too deep; relations nest at most {NESTING} deep"
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
