from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World

from speech2plan.contract import IntentName, Qualifier

# %% what the world offers


@dataclass
class EntityOption:
    """One entity the world contains, in the form the model may refer to it."""

    entity_type: str = field(kw_only=True)
    name: str = field(kw_only=True)
    qualifiers: List[Qualifier] = field(kw_only=True, default_factory=list)

    def render(self) -> str:
        if not self.qualifiers:
            return f"{self.entity_type} '{self.name}'"
        described = ", ".join(f"{q.attribute}={q.value}" for q in self.qualifiers)
        return f"{self.entity_type} '{self.name}' ({described})"


@dataclass
class Vocabulary:
    """The symbols the model is allowed to emit.

    Intents come from hri_actions_msgs. Entities are read from a live world, so
    a room added to the twin is immediately speakable and one removed from it
    stops being speakable.
    """

    intents: List[IntentName] = field(kw_only=True)
    entities: List[EntityOption] = field(kw_only=True)

    @classmethod
    def from_world(cls, world: World) -> Vocabulary:
        entities = [
            EntityOption(
                entity_type=type(room).__name__,
                name=str(room.name),
                qualifiers=[
                    Qualifier(attribute="floor", value=type(room.floor).__name__)
                ],
            )
            for room in world.get_semantic_annotations_by_type(Room)
        ]
        return cls(intents=list(IntentName), entities=entities)

    @property
    def entity_types(self) -> List[str]:
        return sorted({entity.entity_type for entity in self.entities})

    def render(self) -> str:
        intent_lines = "\n".join(f"- {intent}" for intent in self.intents)
        entity_lines = "\n".join(f"- {entity.render()}" for entity in self.entities)
        return f"Intents:\n{intent_lines}\n\nEntities in the world:\n{entity_lines}"
