from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Union

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    Qualifier,
    Relation,
    SpatialRelation,
)
from speech2plan.grounding import Grounder
from speech2plan.placement import (
    PlacementResolver,
    PlacementSnapshot,
    QualifierAttribute,
)

# %% what the world offers

Place = Union[Qualifier, Relation]
"""One thing said about where an entity is."""


@dataclass
class TypeEntry:
    """One entity type the world holds, and what tells its instances apart."""

    entity_type: str = field(kw_only=True)
    count: int = field(kw_only=True, default=1)
    places: List[EntityConstraint] = field(kw_only=True, default_factory=list)
    """Per instance, the shortest description that singles it out, or where it is
    when nothing does. Empty when the world holds the type once, or when the
    instances were not described."""

    def render(self) -> str:
        """One prompt line, as in `- Cup (3): on Table | in Bedroom | 2x in Lounge`."""
        if not self.places:
            counted = f" ({self.count})" if self.count > 1 else ""
            return f"- {self.entity_type}{counted}"
        counted = Counter(
            " ".join(place.place_notation()) or "anywhere" for place in self.places
        )
        described = " | ".join(
            notation if times == 1 else f"{times}x {notation}"
            for notation, times in counted.items()
        )
        return f"- {self.entity_type} ({self.count}): {described}"


@dataclass
class Vocabulary:
    """The symbols the model may emit, and the world they are drawn from."""

    intents: List[IntentName] = field(kw_only=True)
    types: List[TypeEntry] = field(kw_only=True)
    floors: List[str] = field(kw_only=True, default_factory=list)
    """The storeys a `floor` qualifier may name."""

    relations: List[SpatialRelation] = field(
        kw_only=True, default_factory=lambda: list(SpatialRelation)
    )

    @classmethod
    def from_world(cls, world: World, describe_instances: bool = True) -> Vocabulary:
        """Read the entity types out of a world, with what tells instances apart
        unless `describe_instances` is off, which saves the time it takes."""
        snapshot = PlacementSnapshot(world=world)
        describer = InstanceDescriber(
            grounder=Grounder(world=world, qualifier_resolver=snapshot),
            resolver=snapshot,
        )
        by_type = {}
        for entity in describer.grounder.searchable_entities():
            by_type.setdefault(type(entity).__name__, []).append(entity)
        return cls(
            intents=list(IntentName),
            types=[
                TypeEntry(
                    entity_type=entity_type,
                    count=len(entities),
                    places=(
                        [describer.distinguishing(entity) for entity in entities]
                        if describe_instances and len(entities) > 1
                        else []
                    ),
                )
                for entity_type, entities in by_type.items()
            ],
            floors=sorted(
                {
                    type(level).__name__
                    for level in world.get_semantic_annotations_by_type(Level)
                }
            ),
        )

    @property
    def entity_types(self) -> List[str]:
        return sorted(entry.entity_type for entry in self.types)

    def count_of(self, entity_type: str) -> int:
        """How many entities of that type the world holds."""
        for entry in self.types:
            if entry.entity_type == entity_type:
                return entry.count
        return 0

    def render_schema(self) -> str:
        """The vocabulary as prompt text, with the types but none of their instances."""
        return (
            f"{self.render_symbols()}\n\n"
            f"Entity types in the world: {', '.join(self.entity_types)}"
        )

    def render(self) -> str:
        """The vocabulary as prompt text, with what tells instances apart."""
        type_lines = "\n".join(entry.render() for entry in self.types)
        return (
            f"{self.render_symbols()}\n\n"
            "Entity types in the world. Where a type occurs more than once, what "
            "sets each instance apart follows its count:\n"
            f"{type_lines}"
        )

    def render_symbols(self) -> str:
        """The intents and the floors."""
        intent_lines = "\n".join(f"- {intent}" for intent in self.intents)
        return f"Intents:\n{intent_lines}\n\nFloors: {', '.join(self.floors) or 'none'}"


# %% telling instances apart


@dataclass
class InstanceDescriber:
    """Finds the shortest description that singles out one entity of its type."""

    grounder: Grounder = field(kw_only=True)
    resolver: Optional[PlacementResolver] = field(kw_only=True, default=None)

    def __post_init__(self):
        if self.resolver is None:
            self.resolver = PlacementResolver(world=self.grounder.world)

    def distinguishing(self, entity: SemanticAnnotation) -> EntityConstraint:
        """The fewest places that leave `entity` as the only match, or all its
        places when no combination does."""
        places = self.places_of(entity)
        for size in range(1, len(places) + 1):
            for chosen in combinations(places, size):
                constraint = self.constraint_for(entity, list(chosen))
                matched = self.grounder.matches(constraint)
                if len(matched) == 1 and matched[0] is entity:
                    return constraint
        return self.constraint_for(entity, places)

    def places_of(self, entity: SemanticAnnotation) -> List[Place]:
        """Everything that can be said about where the entity is, most telling first."""
        places: List[Place] = []
        if not isinstance(entity, Room):
            room = self.resolver.room_containing(entity)
            if room is not None:
                places.append(self.relation_to(SpatialRelation.IN, room))
            container = self.resolver.container_of(entity)
            if container is not None:
                places.append(self.relation_to(SpatialRelation.IN, container))
            support = self.resolver.support_of(entity)
            if support is not None:
                places.append(self.relation_to(SpatialRelation.ON, support))
        storey = self.resolver.storey_of(entity)
        if storey is not None:
            places.append(Qualifier(attribute=QualifierAttribute.FLOOR, value=storey))
        return places

    @staticmethod
    def relation_to(
        relation: SpatialRelation, reference: SemanticAnnotation
    ) -> Relation:
        return Relation(
            relation=relation,
            reference=EntityConstraint(entity_type=type(reference).__name__),
        )

    @staticmethod
    def constraint_for(
        entity: SemanticAnnotation, places: List[Place]
    ) -> EntityConstraint:
        return EntityConstraint(
            entity_type=type(entity).__name__,
            qualifiers=[place for place in places if isinstance(place, Qualifier)],
            relations=[place for place in places if isinstance(place, Relation)],
        )
