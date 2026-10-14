from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from semantic_digital_twin.semantic_annotations.semantic_annotations import Level, Room
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# %% qualifier names


class QualifierAttribute(StrEnum):
    """Attributes an utterance may use to narrow an entity down."""

    FLOOR = "floor"
    ROOM = "room"
    NAME = "name"


# %% interface


class QualifierResolver(ABC):
    """Decides whether an entity satisfies one qualifier."""

    @abstractmethod
    def satisfies(
        self, entity: SemanticAnnotation, attribute: str, value: str
    ) -> bool:
        """True when the entity matches `attribute=value`."""


# %% placement in the world


@dataclass
class PlacementResolver(QualifierResolver):
    """Resolves `floor` and `room` qualifiers by position.

    The twin relates neither rooms to levels nor objects to rooms, so both are
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

    def storey_of(self, entity: SemanticAnnotation) -> Optional[str]:
        """The highest Level at or below the entity, or None."""
        ceiling = self.elevation_of(entity) + self.storey_tolerance
        below = {
            type(level).__name__: float(level.floor_plane[0].z)
            for level in self.world.get_semantic_annotations_by_type(Level)
            if float(level.floor_plane[0].z) <= ceiling
        }
        if not below:
            return None
        return max(below, key=below.get)

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

    def room_of(self, entity: SemanticAnnotation) -> Optional[str]:
        """The name of the nearest room on the entity's storey, or None."""
        room = self.room_containing(entity)
        return None if room is None else str(room.name)

    def room_named(self, name: str) -> Optional[Room]:
        """The room of that name, or None."""
        for room in self.world.get_semantic_annotations_by_type(Room):
            if str(room.name) == name:
                return room
        return None

    def satisfies(
        self, entity: SemanticAnnotation, attribute: str, value: str
    ) -> bool:
        if attribute == QualifierAttribute.FLOOR:
            return self.storey_of(entity) == value
        if attribute == QualifierAttribute.ROOM:
            return self.room_of(entity) == value
        return False
