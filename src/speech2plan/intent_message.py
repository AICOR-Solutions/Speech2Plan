from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import List, Optional

from hri_actions_msgs.msg import Intent

from speech2plan.contract import ParsedInstruction, Qualifier, ThematicRole
from speech2plan.grounding import (
    Ambiguity,
    GroundingOutcome,
    ResolvedRole,
    Unmatched,
)

# %% payload vocabulary


class PayloadKey(StrEnum):
    """Keys of the JSON object in Intent.data that are not thematic roles.

    The keys of each constraint are the fields of SemanticConstraint.
    """

    OTHER_DATA = "other_data"
    CONSTRAINTS = "constraints"


class GroundingStatus(StrEnum):
    """How far a role got towards a world entity."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    UNFILLED = "unfilled"

    @classmethod
    def of(cls, outcome: GroundingOutcome) -> GroundingStatus:
        """The status a grounding outcome reports."""
        return {
            ResolvedRole: cls.RESOLVED,
            Ambiguity: cls.AMBIGUOUS,
            Unmatched: cls.UNMATCHED,
        }[type(outcome)]


# %% one role


@dataclass
class SemanticConstraint:
    """One thematic role as the plan executive receives it."""

    role: ThematicRole = field(kw_only=True)
    status: GroundingStatus = field(kw_only=True)
    confidence: float = field(kw_only=True, default=0.0)
    entity: Optional[str] = field(kw_only=True, default=None)
    """The world entity the role resolved to."""

    assumed: bool = field(kw_only=True, default=False)
    """True when the entity was the nearest of several, not the only one."""

    entity_type: Optional[str] = field(kw_only=True, default=None)
    qualifiers: List[Qualifier] = field(kw_only=True, default_factory=list)
    candidates: List[str] = field(kw_only=True, default_factory=list)
    """Everything that matched, while the role is still ambiguous."""

    def as_dict(self) -> dict:
        """The record as it appears in the payload."""
        return asdict(self)


# %% the whole intent


@dataclass
class PublishedIntent:
    """A parsed instruction as an hri_actions_msgs/Intent.

    A thematic role is published only once it names a world entity, because a
    consumer reading the standard keys cannot tell an entity named Kitchen from
    the type Kitchen. A role that matched several entities or none appears among
    the constraints instead, which travel with their confidences under the
    message's own `other_data` key, so nothing outside hri_actions_msgs is
    needed to describe the task.
    """

    instruction: ParsedInstruction = field(kw_only=True)
    outcomes: List[GroundingOutcome] = field(kw_only=True)
    priority: int = field(kw_only=True, default=128)
    """128 is the default priority in hri_actions_msgs."""

    def as_message(self) -> Intent:
        message = Intent()
        message.intent = str(self.instruction.intent)
        message.data = self.to_json()
        message.source = self.instruction.speaker
        message.modality = Intent.MODALITY_SPEECH
        message.priority = self.priority
        message.confidence = float(self.instruction.confidence)
        return message

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    def as_dict(self) -> dict:
        """The payload of the `data` field."""
        constraints = self.constraints()
        data = {
            str(constraint.role): constraint.entity
            for constraint in constraints
            if constraint.entity is not None
        }
        data[str(ThematicRole.INPUT)] = self.instruction.utterance
        if self.instruction.locale:
            data[str(ThematicRole.LOCALE)] = self.instruction.locale
        data[str(PayloadKey.OTHER_DATA)] = {
            str(PayloadKey.CONSTRAINTS): [
                constraint.as_dict() for constraint in constraints
            ]
        }
        return data

    def constraints(self) -> List[SemanticConstraint]:
        """One record per role, grounded or not."""
        return [self.constraint_for(outcome) for outcome in self.outcomes] + [
            SemanticConstraint(
                role=role,
                status=GroundingStatus.UNFILLED,
                confidence=self.confidence_of(role),
            )
            for role in self.instruction.unfilled_roles
        ]

    def constraint_for(self, outcome: GroundingOutcome) -> SemanticConstraint:
        """One grounding outcome as a record."""
        constraint = self.instruction.constraint_for(outcome.role)
        resolved = isinstance(outcome, ResolvedRole)
        return SemanticConstraint(
            role=outcome.role,
            status=GroundingStatus.of(outcome),
            confidence=self.confidence_of(outcome.role),
            entity=str(outcome.entity.name) if resolved else None,
            assumed=resolved and outcome.assumed,
            entity_type=constraint.entity_type,
            qualifiers=constraint.qualifiers,
            candidates=[str(entity.name) for entity in outcome.also_matched],
        )

    def confidence_of(self, role: ThematicRole) -> float:
        """What the model scored that role, or zero when it scored nothing."""
        for scored in self.instruction.confidence_per_role:
            if scored.role == role:
                return scored.confidence
        return 0.0
