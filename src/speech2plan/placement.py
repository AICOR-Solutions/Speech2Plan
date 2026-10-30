from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Dict, Optional

from semantic_digital_twin.reasoning.predicates import InsideOf, SupportedBy
from semantic_digital_twin.semantic_annotations.mixins import (
    HasCaseAsRootBody,
    HasRootBody,
    HasSupportingSurface,
    IsStorageSpace,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Container,
    Floor,
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
    ROOM = "room"
    NAME = "name"


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
    """Resolves `floor` and `room` qualifiers, and the `on` and `in` relations.

    The twin relates neither rooms to levels nor objects to rooms, so both are
    derived from elevation and distance. What rests on or lies in what is read off
    the storage spaces of the twin, and failing that off the geometry.
    """

    world: World = field(kw_only=True)
    storey_tolerance: float = field(kw_only=True, default=0.5)
    containment_ratio: float = field(kw_only=True, default=0.9)
    """How much of a thing has to lie within a container's bounds to be in it. A
    thing standing on a box touches its bounds and counts as half inside."""

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

    def satisfies(self, entity: SemanticAnnotation, attribute: str, value: str) -> bool:
        if attribute == QualifierAttribute.FLOOR:
            return self.storey_of(entity) == value
        if attribute == QualifierAttribute.ROOM:
            return self.room_of(entity) == value
        return False

    # %% relations between entities

    def relates(
        self,
        entity: SemanticAnnotation,
        relation: SpatialRelation,
        reference: SemanticAnnotation,
    ) -> bool:
        if relation == SpatialRelation.ON:
            return self.rests_on(entity, reference)
        if isinstance(reference, Room):
            return self.room_containing(entity) is reference
        return self.lies_in(entity, reference)

    def rests_on(self, entity: SemanticAnnotation, support: SemanticAnnotation) -> bool:
        """Whether `entity` stands on `support`, as stored there or by contact."""
        if entity is support or not self.has_body(entity) or not self.has_body(support):
            return False
        if isinstance(support, IsStorageSpace) and not isinstance(
            support, HasCaseAsRootBody
        ):
            if any(stored is entity for stored in support.objects):
                return True
        # Contact needs a shape on both sides, which a perceived object may lack.
        if not entity.root.has_collision() or not support.root.has_collision():
            return False
        return SupportedBy(supported=entity.root, supporting=support.root)()

    def lies_in(
        self, entity: SemanticAnnotation, container: SemanticAnnotation
    ) -> bool:
        """Whether `entity` lies inside `container`, as stored there or by volume.

        Only a container holds anything inside: a cabinet, a drawer, a bowl.
        """
        if (
            entity is container
            or not self.has_body(entity)
            or not self.has_body(container)
        ):
            return False
        if not isinstance(container, (HasCaseAsRootBody, Container)):
            return False
        if isinstance(container, HasCaseAsRootBody) and any(
            stored is entity for stored in container.objects
        ):
            return True
        return InsideOf(
            body=entity.root,
            other=container.root,
            minimum_containment_ratio=self.containment_ratio,
        )()

    def support_of(self, entity: SemanticAnnotation) -> Optional[SemanticAnnotation]:
        """What the entity stands on, a room's floor aside, or None."""
        for support in self.world.get_semantic_annotations_by_type(
            HasSupportingSurface
        ):
            if not isinstance(support, Floor) and self.rests_on(entity, support):
                return support
        return None

    def container_of(self, entity: SemanticAnnotation) -> Optional[SemanticAnnotation]:
        """The container the entity lies in, or None."""
        for container in self.world.get_semantic_annotations_by_type(HasCaseAsRootBody):
            if self.lies_in(entity, container):
                return container
        return None

    @staticmethod
    def has_body(entity: SemanticAnnotation) -> bool:
        """Whether the entity has a body of its own to stand or lie with."""
        return isinstance(entity, HasRootBody) and entity.root is not None


# %% one moment of the world


@dataclass
class PlacementSnapshot(PlacementResolver):
    """A placement resolver that remembers every answer, for a world that holds still.

    Telling instances apart asks the same questions many times over, each of which
    walks the kinematics or checks for contact.
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

    def rests_on(self, entity: SemanticAnnotation, support: SemanticAnnotation) -> bool:
        return self.remembered("on", super().rests_on, entity, support)

    def lies_in(
        self, entity: SemanticAnnotation, container: SemanticAnnotation
    ) -> bool:
        return self.remembered("in", super().lies_in, entity, container)
