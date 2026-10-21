from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import IntentName, Qualifier
from speech2plan.grounding import Grounder
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

    @classmethod
    def from_world(cls, world: World) -> Vocabulary:
        """Read the referable entities out of a world."""
        resolver = PlacementResolver(world=world)
        vocabulary = cls(intents=list(IntentName), entities=[])
        for entity in Grounder(world=world).searchable_entities():
            vocabulary.entities.append(
                EntityOption(
                    entity_type=type(entity).__name__,
                    name=str(entity.name),
                    qualifiers=cls.qualifiers_for(entity, resolver),
                )
            )
        return vocabulary

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
