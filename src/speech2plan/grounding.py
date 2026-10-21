from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, Union

from semantic_digital_twin.grasping.grasp_candidates import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Agent,
    Furniture,
    Handle,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import EntityConstraint, ParsedInstruction, ThematicRole
from speech2plan.placement import (
    PlacementResolver,
    QualifierAttribute,
    QualifierResolver,
)

# %% outcomes


@dataclass
class ResolvedRole:
    """Exactly one world entity matched the constraint filling a role."""

    role: ThematicRole = field(kw_only=True)
    entity: SemanticAnnotation = field(kw_only=True)
    alternatives: List[SemanticAnnotation] = field(kw_only=True, default_factory=list)
    """The entities that also matched, when the speaker's floor decided it."""

    @property
    def assumed(self) -> bool:
        """Whether the speaker's words left this open."""
        return bool(self.alternatives)

    @property
    def also_matched(self) -> List[SemanticAnnotation]:
        """What else matched the constraint."""
        return self.alternatives


@dataclass
class Ambiguity:
    """Several world entities matched one constraint."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)
    candidates: List[SemanticAnnotation] = field(kw_only=True)

    @property
    def also_matched(self) -> List[SemanticAnnotation]:
        """What else matched the constraint."""
        return self.candidates


@dataclass
class Unmatched:
    """No world entity matched the constraint."""

    role: ThematicRole = field(kw_only=True)
    constraint: EntityConstraint = field(kw_only=True)

    @property
    def also_matched(self) -> List[SemanticAnnotation]:
        """What else matched the constraint, which is nothing."""
        return []


GroundingOutcome = Union[ResolvedRole, Ambiguity, Unmatched]


# %% what to assume when the words leave several


class CandidatePreference(ABC):
    """Narrows the entities that matched to the ones a speaker likely meant."""

    @abstractmethod
    def narrow(self, candidates: List[SemanticAnnotation]) -> List[SemanticAnnotation]:
        """The candidates worth considering, out of the ones that matched."""


@dataclass
class NoAssumption(CandidatePreference):
    """Assumes nothing, so every entity that matched stays in question."""

    def narrow(self, candidates: List[SemanticAnnotation]) -> List[SemanticAnnotation]:
        return candidates


@dataclass
class AgentStoreyPreference(CandidatePreference):
    """Prefers the candidates on the storey the agents of the world stand on.

    A person who says "the kitchen" in a two kitchen building means the one
    where they are, unless they say otherwise.
    """

    world: World = field(kw_only=True)
    resolver: Optional[PlacementResolver] = field(kw_only=True, default=None)

    def __post_init__(self):
        if self.resolver is None:
            self.resolver = PlacementResolver(world=self.world)

    def storey(self) -> Optional[str]:
        """The storey every agent in the world is on, or None."""
        storeys = {
            self.resolver.storey_of(agent)
            for agent in self.world.get_semantic_annotations_by_type(Agent)
        }
        return storeys.pop() if len(storeys) == 1 else None

    def narrow(self, candidates: List[SemanticAnnotation]) -> List[SemanticAnnotation]:
        """The candidates on that storey, or all of them when none is there."""
        storey = self.storey()
        if storey is None:
            return candidates
        near = [
            candidate
            for candidate in candidates
            if self.resolver.storey_of(candidate) == storey
        ]
        return near or candidates


# %% grounder


@dataclass
class Grounder:
    """Maps entity constraints onto world entities."""

    world: World = field(kw_only=True)
    qualifier_resolver: Optional[QualifierResolver] = field(kw_only=True, default=None)
    searchable_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Room, HasGraspCandidates, Furniture)
    )
    """Annotation bases an utterance may refer to."""

    excluded_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Handle,)
    )
    """Parts of something else, which an utterance never names on their own."""

    preference: Optional[CandidatePreference] = field(kw_only=True, default=None)
    """Which candidate to assume when the words leave several."""

    def __post_init__(self):
        if self.qualifier_resolver is None:
            self.qualifier_resolver = PlacementResolver(world=self.world)
        if self.preference is None:
            self.preference = AgentStoreyPreference(world=self.world)

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
                if not isinstance(entity, self.excluded_types):
                    found[str(entity.name)] = entity
        return list(found.values())

    def entity_named(self, name: str) -> Optional[SemanticAnnotation]:
        """The entity with that name, or None."""
        for entity in self.searchable_entities():
            if str(entity.name) == name:
                return entity
        return None

    def entities_of_type(self, entity_type: str) -> List[SemanticAnnotation]:
        """Everything of that type the world holds, qualifiers ignored."""
        return [
            entity
            for entity in self.searchable_entities()
            if type(entity).__name__ == entity_type
        ]

    def satisfies(self, entity: SemanticAnnotation, qualifier) -> bool:
        """True when the entity matches one qualifier."""
        if qualifier.attribute == QualifierAttribute.NAME:
            return str(entity.name) == qualifier.value
        return self.qualifier_resolver.satisfies(
            entity, qualifier.attribute, qualifier.value
        )

    def ground_constraint(
        self, role: ThematicRole, constraint: EntityConstraint
    ) -> GroundingOutcome:
        """Resolve one constraint against the world."""
        candidates = [
            entity
            for entity in self.entities_of_type(constraint.entity_type)
            if all(
                self.satisfies(entity, qualifier) for qualifier in constraint.qualifiers
            )
        ]
        if not candidates:
            return Unmatched(role=role, constraint=constraint)
        if len(candidates) == 1:
            return ResolvedRole(role=role, entity=candidates[0])
        nearest = self.preference.narrow(candidates)
        if len(nearest) == 1:
            return ResolvedRole(
                role=role,
                entity=nearest[0],
                alternatives=[
                    candidate for candidate in candidates if candidate is not nearest[0]
                ],
            )
        return Ambiguity(role=role, constraint=constraint, candidates=nearest)
