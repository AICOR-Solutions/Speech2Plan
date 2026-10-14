from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, Union

from semantic_digital_twin.grasping.grasp_candidates import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Furniture,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import EntityConstraint, ParsedInstruction, ThematicRole
from speech2plan.placement import PlacementResolver, QualifierResolver

# %% outcomes


@dataclass
class ResolvedRole:
    """Exactly one world entity matched the constraint filling a role."""

    role: ThematicRole = field(kw_only=True)
    entity: SemanticAnnotation = field(kw_only=True)


@dataclass
class Ambiguity:
    """Several world entities matched one constraint."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)
    candidates: List[SemanticAnnotation] = field(kw_only=True)


@dataclass
class Unmatched:
    """No world entity matched the constraint."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)


GroundingOutcome = Union[ResolvedRole, Ambiguity, Unmatched]


# %% grounder


@dataclass
class Grounder:
    """Maps entity constraints onto world entities."""

    world: World = field(kw_only=True)
    qualifier_resolver: Optional[QualifierResolver] = field(kw_only=True, default=None)
    searchable_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Room, HasGraspCandidates, Furniture)
    )

    def __post_init__(self):
        if self.qualifier_resolver is None:
            self.qualifier_resolver = PlacementResolver(world=self.world)

    def ground(self, instruction: ParsedInstruction) -> List[GroundingOutcome]:
        """One outcome per filled role."""
        return [
            self.ground_constraint(binding.role, binding.constraint)
            for binding in instruction.roles
        ]

    def searchable_entities(self) -> List[SemanticAnnotation]:
        """Every annotation a constraint may match, without duplicates."""
        found = {}
        for searchable in self.searchable_types:
            for entity in self.world.get_semantic_annotations_by_type(searchable):
                found[str(entity.name)] = entity
        return list(found.values())

    def ground_constraint(
        self, role: ThematicRole, constraint: EntityConstraint
    ) -> GroundingOutcome:
        """Resolve one constraint against the world."""
        candidates = [
            entity
            for entity in self.searchable_entities()
            if type(entity).__name__ == constraint.entity_type
            and all(
                self.qualifier_resolver.satisfies(
                    entity, qualifier.attribute, qualifier.value
                )
                for qualifier in constraint.qualifiers
            )
        ]
        if not candidates:
            return Unmatched(role=role, constraint=constraint)
        if len(candidates) == 1:
            return ResolvedRole(role=role, entity=candidates[0])
        return Ambiguity(role=role, constraint=constraint, candidates=candidates)
