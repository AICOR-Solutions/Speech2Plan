from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional

from hri_actions_msgs.msg import Intent

# %% vocabulary, taken from the ROS4HRI message


class IntentName(StrEnum):
    """The intents of hri_actions_msgs/Intent.

    Values come from the message itself so there is one definition. Speech2Plan
    recognises the subset below; the message defines more.
    """

    RAW_USER_INPUT = Intent.RAW_USER_INPUT
    MOVE_TO = Intent.MOVE_TO
    GUIDE = Intent.GUIDE
    GRAB_OBJECT = Intent.GRAB_OBJECT
    BRING_OBJECT = Intent.BRING_OBJECT
    PLACE_OBJECT = Intent.PLACE_OBJECT


class ThematicRole(StrEnum):
    """Keys of the Intent `data` object.

    `object` is the entity undergoing the intent, `goal` the entity it moves
    towards, `recipient` the entity that receives the object.
    """

    AGENT = "agent"
    OBJECT = "object"
    GOAL = "goal"
    RECIPIENT = "recipient"
    INPUT = "input"
    LOCALE = "locale"


# %% entity references


@dataclass
class Qualifier:
    """One attribute that narrows which entity is meant, such as a floor."""

    attribute: str = field(kw_only=True)
    value: str = field(kw_only=True)


@dataclass
class EntityConstraint:
    """A description of an entity, never a single instance.

    The language model emits these. Choosing between the instances that match
    is the grounder's job, so no room number is ever invented.
    """

    entity_type: str = field(kw_only=True)
    """Name of a semantic annotation class, validated against the vocabulary."""

    qualifiers: List[Qualifier] = field(kw_only=True, default_factory=list)

    spoken_form: Optional[str] = field(kw_only=True, default=None)
    """The words the speaker used, kept for clarification questions."""


@dataclass
class RoleBinding:
    """One thematic role and the entity that fills it."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)


# %% the parse result


@dataclass
class ParsedInstruction:
    """What one utterance means, in the vocabulary of hri_actions_msgs/Intent.

    A role the speaker left open is absent from `roles` and named in
    `unfilled_roles`. "Bring this to the kitchen" has a goal and no object
    until gaze or a question supplies one.
    """

    utterance: str = field(kw_only=True)
    speaker: str = field(kw_only=True)
    """ROS4HRI person id, or Intent.UNKNOWN_AGENT when no person is bound."""

    intent: IntentName = field(kw_only=True)
    roles: List[RoleBinding] = field(kw_only=True, default_factory=list)
    unfilled_roles: List[ThematicRole] = field(kw_only=True, default_factory=list)
    confidence: float = field(kw_only=True, default=0.0)
    locale: Optional[str] = field(kw_only=True, default=None)

    def constraint_for(self, role: ThematicRole) -> Optional[EntityConstraint]:
        for binding in self.roles:
            if binding.role == role:
                return binding.constraint
        return None
