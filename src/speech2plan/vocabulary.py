from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Type

from semantic_digital_twin.grasping.grasp_candidates import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Furniture,
    Handle,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import IntentName, Qualifier
from speech2plan.placement import PlacementResolver, QualifierAttribute

# %% what the world offers


@dataclass
class EntityOption:
    """One entity the world contains, as the model may refer to it."""

    entity_type: str = field(kw_only=True)
    name: str = field(kw_only=True)
    qualifiers: List[Qualifier] = field(kw_only=True, default_factory=list)

    def render(self) -> str:
        """One prompt line describing this entity."""
        if not self.qualifiers:
            return f"{self.entity_type} '{self.name}'"
        described = ", ".join(f"{q.attribute}={q.value}" for q in self.qualifiers)
        return f"{self.entity_type} '{self.name}' ({described})"


@dataclass
class Vocabulary:
    """The symbols the model may emit."""

    intents: List[IntentName] = field(kw_only=True)
    entities: List[EntityOption] = field(kw_only=True)

    searchable_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Room, HasGraspCandidates, Furniture)
    )
    """Annotation bases an utterance may refer to."""

    excluded_types: Tuple[Type[SemanticAnnotation], ...] = field(
        kw_only=True, default=(Handle,)
    )
    """Parts of something else, which an utterance never names on their own."""

    @classmethod
    def from_world(cls, world: World) -> Vocabulary:
        """Read the referable entities out of a world."""
        resolver = PlacementResolver(world=world)
        vocabulary = cls(intents=list(IntentName), entities=[])
        for entity in vocabulary.referable_entities(world):
            vocabulary.entities.append(
                EntityOption(
                    entity_type=type(entity).__name__,
                    name=str(entity.name),
                    qualifiers=cls.qualifiers_for(entity, resolver),
                )
            )
        return vocabulary

    def referable_entities(self, world: World) -> List[SemanticAnnotation]:
        """Every annotation an utterance may refer to, without duplicates."""
        found = {}
        for searchable in self.searchable_types:
            for entity in world.get_semantic_annotations_by_type(searchable):
                if not isinstance(entity, self.excluded_types):
                    found[str(entity.name)] = entity
        return list(found.values())

    @staticmethod
    def qualifiers_for(
        entity: SemanticAnnotation, resolver: PlacementResolver
    ) -> List[Qualifier]:
        """The qualifiers that distinguish this entity from its namesakes."""
        if isinstance(entity, Room):
            storey = resolver.storey_of(entity)
            if storey is None:
                return []
            return [Qualifier(attribute=QualifierAttribute.FLOOR, value=storey)]
        room = resolver.room_of(entity)
        if room is None:
            return []
        return [Qualifier(attribute=QualifierAttribute.ROOM, value=room)]

    @property
    def entity_types(self) -> List[str]:
        return sorted({entity.entity_type for entity in self.entities})

    def render(self) -> str:
        """The vocabulary as prompt text."""
        intent_lines = "\n".join(f"- {intent}" for intent in self.intents)
        entity_lines = "\n".join(f"- {entity.render()}" for entity in self.entities)
        return f"Intents:\n{intent_lines}\n\nEntities in the world:\n{entity_lines}"
