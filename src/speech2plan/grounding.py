from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type, Union

from krrood.entity_query_language import factories as eql
from krrood.entity_query_language.predicate import symbolic_function
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Agent,
    Furniture,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import (
    EntityConstraint,
    ParsedInstruction,
    Qualifier,
    Relation,
    ThematicRole,
)
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


# %% the conditions of a query


@symbolic_function
def satisfies(grounder: Grounder, entity: SemanticAnnotation, qualifier: Qualifier):
    """Whether the entity matches one qualifier, evaluated when the query runs."""
    return grounder.satisfies(entity, qualifier)


@symbolic_function
def relates(
    grounder: Grounder,
    entity: SemanticAnnotation,
    relation: Relation,
    reference: SemanticAnnotation,
):
    """Whether the entity stands in the relation to the reference."""
    return grounder.qualifier_resolver.relates(entity, relation.relation, reference)


# %% grounder


@dataclass
class Grounder:
    """Maps the descriptions of places onto the rooms and furniture of the world.

    Only places are grounded: what the robot carries is put on its tray by people, so
    it is never looked for. A constraint becomes an entity query: one variable over the instances of its
    type, filtered by its qualifiers, and one more variable per relation, ranging
    over the instances of the reference's type and filtered by the reference's own
    constraint, so a description may nest as deep as the speaker's words did.
    """

    world: World = field(kw_only=True)
    qualifier_resolver: Optional[QualifierResolver] = field(kw_only=True, default=None)
    searchable_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Room, Furniture)
    )
    """Annotation bases an utterance may name as a place."""

    preference: Optional[CandidatePreference] = field(kw_only=True, default=None)
    """Which candidate to assume when the words leave several."""

    def __post_init__(self):
        if self.qualifier_resolver is None:
            self.qualifier_resolver = PlacementResolver(world=self.world)
        if self.preference is None:
            self.preference = AgentStoreyPreference(world=self.world)

    def ground(self, instruction: ParsedInstruction) -> List[GroundingOutcome]:
        """One outcome per filled role that names a place."""
        return [
            self.ground_constraint(binding.role, binding.constraint)
            for binding in instruction.places
        ]

    def searchable_entities(self) -> List[SemanticAnnotation]:
        """Every annotation a constraint may match, without duplicates."""
        found = {}
        for searchable in self.searchable_types:
            for entity in self.world.get_semantic_annotations_by_type(searchable):
                found[str(entity.name)] = entity
        return list(found.values())

    def entities_of_type(self, entity_type: Optional[str]) -> List[SemanticAnnotation]:
        """Everything of that type the world holds, qualifiers ignored."""
        return [
            entity
            for entity in self.searchable_entities()
            if type(entity).__name__ == entity_type
        ]

    def satisfies(self, entity: SemanticAnnotation, qualifier: Qualifier) -> bool:
        """True when the entity matches one qualifier."""
        if qualifier.attribute == QualifierAttribute.NAME:
            return str(entity.name) == qualifier.value
        return self.qualifier_resolver.satisfies(
            entity, qualifier.attribute, qualifier.value
        )

    # %% constraints as entity queries

    def matches(self, constraint: EntityConstraint) -> List[SemanticAnnotation]:
        """Every entity the constraint describes, in the order the world holds them."""
        described = self.variable_for(constraint)
        if described is None:
            return []
        conditions = self.conditions(described, constraint)
        if conditions is None:
            return []
        query = eql.entity(described)
        if conditions:
            query = query.where(*conditions)
        # A relation joins over its references, so an entity comes once per reference
        # it relates to. Rooms of one type compare and hash alike, which rules out
        # distinct(); identity tells them apart.
        found = {id(match) for match in eql.an(query).evaluate()}
        return [
            candidate
            for candidate in self.entities_of_type(constraint.entity_type)
            if id(candidate) in found
        ]

    def variable_for(self, constraint: EntityConstraint):
        """A query variable over the instances of the constraint's type, or None
        when the world holds none."""
        instances = self.entities_of_type(constraint.entity_type)
        if not instances:
            return None
        return eql.variable(type(instances[0]), domain=instances)

    def conditions(self, described, constraint: EntityConstraint) -> Optional[list]:
        """The conditions on `described` that the constraint states, or None when one
        of them names a kind of thing the world does not hold."""
        conditions = [
            satisfies(self, described, qualifier) for qualifier in constraint.qualifiers
        ]
        for relation in constraint.relations:
            reference = self.variable_for(relation.reference)
            reference_conditions = (
                None
                if reference is None
                else self.conditions(reference, relation.reference)
            )
            if reference_conditions is None:
                return None
            conditions.append(relates(self, described, relation, reference))
            conditions.extend(reference_conditions)
        return conditions

    def ground_constraint(
        self, role: ThematicRole, constraint: EntityConstraint
    ) -> GroundingOutcome:
        """Resolve one constraint against the world."""
        candidates = self.matches(constraint)
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
