from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    Cup,
    FirstFloor,
    Floor,
    GroundFloor,
    Kitchen,
    LivingRoom,
    Plate,
    Room,
    Table,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

# %% interface


class WorldSource(ABC):
    """Where the world the parser grounds against comes from."""

    @abstractmethod
    def load(self) -> World:
        """Return a world."""

    def is_available(self) -> bool:
        """True when `load` can succeed now."""
        return True


# %% prototyping


@dataclass
class PrototypeWard(WorldSource):
    """A two storey building with a kitchen on each floor and a few objects."""

    storey_height: float = field(kw_only=True, default=3.0)
    room_extent: float = field(kw_only=True, default=3.0)

    def load(self) -> World:
        world = World.create_with_root_body("root")
        with world.modify_world():
            self.add_storey(world, GroundFloor, "ground_floor", 0.0)
            self.add_storey(world, FirstFloor, "first_floor", self.storey_height)
            world.add_semantic_annotations(
                [
                    self.add_room(world, Kitchen, "kitchen_ground", 0.0, 0.0),
                    self.add_room(world, LivingRoom, "living_room", 5.0, 0.0),
                    self.add_room(
                        world, Kitchen, "kitchen_first", 0.0, self.storey_height
                    ),
                    self.add_room(world, Bathroom, "bathroom", 5.0, self.storey_height),
                ]
            )
            self.add_object(world, Table, "table", 1.0, 0.0)
            self.add_object(world, Cup, "cup", 1.0, 0.0, height=0.8)
            self.add_object(world, Plate, "plate", 1.2, 0.0, height=0.8)
            self.add_object(world, Cup, "second_cup", 1.0, self.storey_height, height=0.8)
        return world

    def add_object(
        self,
        world: World,
        object_type,
        name: str,
        offset: float,
        elevation: float,
        height: float = 0.0,
    ):
        """Add a graspable object standing at `offset` on the given storey."""
        return object_type.create_with_new_body_in_world(
            name=name,
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=offset, y=1.0, z=elevation + height
            ),
            scale=Scale(0.1, 0.1, 0.1),
        )

    def add_storey(self, world: World, level_type, name: str, elevation: float):
        """Add a Level spanning the building at `elevation`."""
        return level_type.create_with_new_region_in_world(
            name=name,
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=elevation + self.storey_height / 2
            ),
            scale=Scale(20, 20, self.storey_height),
        )

    def add_room(
        self, world: World, room_type, name: str, offset: float, elevation: float
    ) -> Room:
        """Add a square room of `room_type` with its own floor surface."""
        extent = self.room_extent
        floor = Floor.create_with_new_body_from_polytope_in_world(
            name=f"{name}_floor",
            world=world,
            floor_polytope=[
                Point3(0, 0, 0),
                Point3(0, extent, 0),
                Point3(extent, extent, 0),
                Point3(extent, 0, 0),
            ],
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=offset, y=0, z=elevation
            ),
        )
        return room_type(floor=floor, name=PrefixedName(name))
