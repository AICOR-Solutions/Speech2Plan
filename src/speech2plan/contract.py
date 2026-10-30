from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from hri_actions_msgs.msg import Intent

# %% vocabulary


class IntentName(StrEnum):
    """The subset of hri_actions_msgs/Intent that Speech2Plan recognises."""

    RAW_USER_INPUT = Intent.RAW_USER_INPUT
    MOVE_TO = Intent.MOVE_TO
    GUIDE = Intent.GUIDE
    GRAB_OBJECT = Intent.GRAB_OBJECT
    BRING_OBJECT = Intent.BRING_OBJECT
    PLACE_OBJECT = Intent.PLACE_OBJECT


class ThematicRole(StrEnum):
    """Keys of the Intent `data` object."""

    AGENT = "agent"
    OBJECT = "object"
    GOAL = "goal"
    RECIPIENT = "recipient"
    INPUT = "input"
    LOCALE = "locale"


# %% entity references


@dataclass
class Qualifier:
    """An attribute and value narrowing which entity is meant."""

    attribute: str = field(kw_only=True)
    value: str = field(kw_only=True)


UNNAMED_TYPE = "Thing"
"""The type of a thing the speaker leaves unnamed, as in 'the thing on the sofa',
which any movable thing fits."""


class SpatialRelation(StrEnum):
    """How an entity is placed relative to another one."""

    ON = "on"
    """Rests on the other, as a cup on a table."""

    IN = "in"
    """Lies inside the other, a room or a container such as a cabinet."""


@dataclass
class EntityConstraint:
    """A description of an entity, which the grounder resolves to instances."""

    entity_type: Optional[str] = field(kw_only=True)
    """The type, or None when the speaker named a kind of thing the world lacks."""

    qualifiers: List[Qualifier] = field(kw_only=True, default_factory=list)
    relations: List[Relation] = field(kw_only=True, default_factory=list)
    """Where the entity is relative to other entities, each described in turn."""

    spoken_form: Optional[str] = field(kw_only=True, default=None)
    """The words the speaker used."""

    def notation(self) -> str:
        """The constraint in the compact form the model reads, as in
        `Cup on Table in (Kitchen floor=GroundFloor)`."""
        return " ".join([self.entity_type or "?", *self.place_notation()]).strip()

    def place_notation(self) -> List[str]:
        """The qualifiers and relations alone, without the type."""
        return [
            f"{relation.relation} {relation.reference.reference_notation()}"
            for relation in self.relations
        ] + [
            f"{qualifier.attribute}={qualifier.value}" for qualifier in self.qualifiers
        ]

    def reference_notation(self) -> str:
        """The notation, parenthesised when it is more than a type."""
        if self.qualifiers or self.relations:
            return f"({self.notation()})"
        return self.notation()


@dataclass
class Relation:
    """An entity placed relative to another entity, which is described in turn."""

    relation: SpatialRelation = field(kw_only=True)
    reference: EntityConstraint = field(kw_only=True)


@dataclass
class RoleBinding:
    """A thematic role and the entity description filling it."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)


@dataclass
class RoleConfidence:
    """How sure the model is about one role."""

    role: ThematicRole = field(kw_only=True)
    confidence: float = field(kw_only=True)


# %% the parse result


@dataclass
class ParsedInstruction:
    """What one utterance means, in the vocabulary of hri_actions_msgs/Intent."""

    utterance: str = field(kw_only=True)
    speaker: str = field(kw_only=True)
    """ROS4HRI person id, or Intent.UNKNOWN_AGENT."""

    intent: IntentName = field(kw_only=True)
    roles: List[RoleBinding] = field(kw_only=True, default_factory=list)
    unfilled_roles: List[ThematicRole] = field(kw_only=True, default_factory=list)
    intent_confidence: float = field(kw_only=True, default=0.0)
    confidence_per_role: List[RoleConfidence] = field(
        kw_only=True, default_factory=list
    )
    locale: Optional[str] = field(kw_only=True, default=None)

    @property
    def confidence(self) -> float:
        """The weakest confidence in the parse."""
        return min(
            [self.intent_confidence]
            + [scored.confidence for scored in self.confidence_per_role]
        )

    def constraint_for(self, role: ThematicRole) -> Optional[EntityConstraint]:
        """The constraint filling `role`, or None when it is unfilled."""
        for binding in self.roles:
            if binding.role == role:
                return binding.constraint
        return None
