from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World

from speech2plan.contract import (
    EntityConstraint,
    ParsedInstruction,
    ThematicRole,
)

# %% outcomes


@dataclass
class ResolvedRole:
    """Exactly one world entity matched the constraint filling a role."""

    role: ThematicRole = field(kw_only=True)
    entity: Room = field(kw_only=True)


@dataclass
class Ambiguity:
    """Several world entities matched one constraint.

    This is not model uncertainty. The parse was fine and the world holds more
    than one match, so the answer is a question to the speaker, not a retry.
    """

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)
    candidates: List[Room] = field(kw_only=True)


@dataclass
class Unmatched:
    """No world entity matched the constraint."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)


GroundingOutcome = Union[ResolvedRole, Ambiguity, Unmatched]


# %% grounder


@dataclass
class Grounder:
    """Maps constraints onto world entities.

    Deterministic on purpose: the same world and constraint always give the
    same outcome, which makes the ambiguity cases testable without a model.
    """

    world: World = field(kw_only=True)

    def ground(self, instruction: ParsedInstruction) -> List[GroundingOutcome]:
        return [
            self.ground_constraint(binding.role, binding.constraint)
            for binding in instruction.roles
        ]

    def ground_constraint(
        self, role: ThematicRole, constraint: EntityConstraint
    ) -> GroundingOutcome:
        candidates = [
            room
            for room in self.world.get_semantic_annotations_by_type(Room)
            if type(room).__name__ == constraint.entity_type
            and self.satisfies_qualifiers(room, constraint)
        ]
        if not candidates:
            return Unmatched(role=role, constraint=constraint)
        if len(candidates) == 1:
            return ResolvedRole(role=role, entity=candidates[0])
        return Ambiguity(role=role, constraint=constraint, candidates=candidates)

    def satisfies_qualifiers(self, room: Room, constraint: EntityConstraint) -> bool:
        return all(
            qualifier.attribute == "floor"
            and type(room.floor).__name__ == qualifier.value
            for qualifier in constraint.qualifiers
        )
