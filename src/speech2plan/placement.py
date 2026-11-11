from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Dict, Optional

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Room,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import SpatialRelation

# %% qualifier names


class QualifierAttribute(StrEnum):
    """Attributes an utterance may use to narrow an entity down."""

    FLOOR = "floor"
    NUMBER = "number"
    """The number a place carries, as people read it off the door: 2.58, T.0.4.1."""

    NAME = "name"


def number_of(entity: SemanticAnnotation) -> Optional[str]:
    """The number the entity carries, which is its name when that reads like the
    number on a door: digits and letters split by dots, no words such as
    'elevator_room'."""
    name = entity.name.name
    if re.fullmatch(r"[A-Za-z0-9.]+", name) and any(c.isdigit() for c in name):
        return name
    return None


# %% interface


class QualifierResolver(ABC):
    """Decides whether an entity satisfies one qualifier."""

    @abstractmethod
    def satisfies(self, entity: SemanticAnnotation, attribute: str, value: str) -> bool:
        """True when the entity matches `attribute=value`."""

    def relates(
        self,
        entity: SemanticAnnotation,
        relation: SpatialRelation,
        reference: SemanticAnnotation,
    ) -> bool:
        """True when the entity stands in `relation` to `reference`."""
        return False


# %% placement in the world


@dataclass
class PlacementResolver(QualifierResolver):
    """Resolves the `floor` and `number` qualifiers, and the `in` relation of a place
    to a room.

    The twin relates neither rooms to levels nor furniture to rooms, so both are
    derived from elevation and distance.
    """

    world: World = field(kw_only=True)
    storey_tolerance: float = field(kw_only=True, default=0.5)

    def body_of(self, entity: SemanticAnnotation) -> SemanticAnnotation:
        """The body an entity is placed by, which for a room is its floor."""
        return entity.floor.root if isinstance(entity, Room) else entity.root

    def elevation_of(self, entity: SemanticAnnotation) -> float:
        """The entity's height above the world root."""
        return float(self.pose_of(entity).position.z)

    def pose_of(self, entity: SemanticAnnotation):
        """The entity's pose in the world root frame."""
        return self.world.compute_forward_kinematics(
            self.world.root, self.body_of(entity)
        )

    def level_of(self, entity: SemanticAnnotation) -> Optional[Level]:
        """The highest Level at or below the entity, or None."""
        ceiling = self.elevation_of(entity) + self.storey_tolerance
        below = [
            level
            for level in self.world.get_semantic_annotations_by_type(Level)
            if float(level.floor_plane[0].z) <= ceiling
        ]
        if not below:
            return None
        return max(below, key=lambda level: float(level.floor_plane[0].z))

    def storey_of(self, entity: SemanticAnnotation) -> Optional[str]:
        """The name of the Level type at or below the entity, or None."""
        level = self.level_of(entity)
        return None if level is None else type(level).__name__

    def room_containing(self, entity: SemanticAnnotation) -> Optional[Room]:
        """The nearest room on the entity's storey, or None."""
        storey = self.storey_of(entity)
        pose = self.pose_of(entity).position
        distances = {
            room: (float(self.pose_of(room).position.x) - float(pose.x)) ** 2
            + (float(self.pose_of(room).position.y) - float(pose.y)) ** 2
            for room in self.world.get_semantic_annotations_by_type(Room)
            if self.storey_of(room) == storey
        }
        if not distances:
            return None
        return min(distances, key=distances.get)

    def satisfies(self, entity: SemanticAnnotation, attribute: str, value: str) -> bool:
        if attribute == QualifierAttribute.FLOOR:
            return self.storey_of(entity) == value
        if attribute == QualifierAttribute.NUMBER:
            return number_of(entity) == value
        return False

    # %% relations between places

    def relates(
        self,
        entity: SemanticAnnotation,
        relation: SpatialRelation,
        reference: SemanticAnnotation,
    ) -> bool:
        return isinstance(reference, Room) and self.room_containing(entity) is reference


# %% one moment of the world


@dataclass
class PlacementSnapshot(PlacementResolver):
    """A placement resolver that remembers every answer, for a world that holds still.

    Telling instances apart asks the same questions many times over, each of which
    walks the kinematics.
    """

    answers: Dict[tuple, object] = field(kw_only=True, default_factory=dict)

    def remembered(self, question: str, compute: Callable, *entities):
        key = (question, *(id(entity) for entity in entities))
        if key not in self.answers:
            self.answers[key] = compute(*entities)
        return self.answers[key]

    def storey_of(self, entity: SemanticAnnotation) -> Optional[str]:
        return self.remembered("storey", super().storey_of, entity)

    def room_containing(self, entity: SemanticAnnotation) -> Optional[Room]:
        return self.remembered("room", super().room_containing, entity)
