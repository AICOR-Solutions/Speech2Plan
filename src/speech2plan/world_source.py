from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    Bedroom,
    Cabinet,
    CoffeeTable,
    CounterTop,
    Cup,
    FirstFloor,
    Floor,
    GroundFloor,
    Human,
    Kitchen,
    LivingRoom,
    Room,
    SideTable,
    Sofa,
    Table,
    Wardrobe,
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
    """A two storey building with a kitchen on each floor, a living room, a bathroom
    and a table.

    The nurse stands on the ground floor, which is what makes "the kitchen" resolve
    without asking. The cup on the table is one of the things lying about, which are
    no places.
    """

    storey_height: float = field(kw_only=True, default=3.0)
    footprint: float = field(kw_only=True, default=20.0)
    """The side length of the square each storey spans."""

    room_extent: float = field(kw_only=True, default=3.0)

    def load(self) -> World:
        world = World.create_with_root_body("root")
        with world.modify_world():
            self.add_storeys(world)
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
            self.add_furniture(
                world, [(Table, "table", 1.2, 1.0, 0.0, Scale(0.6, 0.6, 0.75))]
            )
            self.add_box(world, Cup, "cup", Point3(1.0, 1.0, 0.8), Scale(0.1, 0.1, 0.1))
            self.add_box(
                world, Human, "nurse", Point3(0.5, 1.0, 0.0), Scale(0.1, 0.1, 0.1)
            )
        return world

    def add_storeys(self, world: World) -> None:
        """Add a Level region spanning each storey."""
        for level_type, name, elevation in (
            (GroundFloor, "ground_floor", 0.0),
            (FirstFloor, "first_floor", self.storey_height),
        ):
            level_type.create_with_new_region_in_world(
                name=name,
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=elevation + self.storey_height / 2
                ),
                scale=Scale(self.footprint, self.footprint, self.storey_height),
            )

    def add_box(
        self, world: World, annotation_type, name: str, position: Point3, scale: Scale
    ):
        """Add an annotation with a box body of `scale` centred at `position`."""
        return annotation_type.create_with_new_body_in_world(
            name=name,
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(position.x), y=float(position.y), z=float(position.z)
            ),
            scale=scale,
        )

    def add_furniture(self, world: World, placements: list) -> None:
        """Add furniture per (annotation type, name, x, y, elevation, scale) placement,
        standing on the floor at `elevation`."""
        for annotation_type, name, x, y, elevation, scale in placements:
            self.add_box(
                world,
                annotation_type,
                name,
                Point3(x, y, elevation + scale.z / 2),
                scale,
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


@dataclass
class CareWard(PrototypeWard):
    """The prototype ward with a patient room, a lounge, and furniture in its rooms.

    The two living rooms and the two tables and sofas lie on different floors, so the
    floor the nurse stands on settles them unless the speaker names another.
    """

    def load(self) -> World:
        world = super().load()
        first = self.storey_height
        with world.modify_world():
            world.add_semantic_annotations(
                [
                    self.add_room(world, Bedroom, "patient_room", -5.0, 0.0),
                    self.add_room(world, LivingRoom, "lounge", -5.0, first),
                ]
            )
            self.add_furniture(
                world,
                [
                    (CounterTop, "kitchen_counter", 2.0, 1.0, 0.0, Scale(0.8, 0.6, 0.9)),
                    (Sofa, "sofa", 6.0, 2.0, 0.0, Scale(1.6, 0.8, 0.45)),
                    (CoffeeTable, "coffee_table", 6.5, 1.0, 0.0, Scale(0.8, 0.6, 0.45)),
                    (Wardrobe, "wardrobe", -4.0, 2.5, 0.0, Scale(1.0, 0.6, 2.0)),
                    (SideTable, "bedside_table", -3.0, 1.0, 0.0, Scale(0.6, 0.4, 0.55)),
                    (Table, "first_floor_table", 1.4, 1.0, first, Scale(1.0, 0.6, 0.75)),
                    (Cabinet, "bathroom_cabinet", 6.0, 1.0, first, Scale(0.8, 0.5, 1.0)),
                    (Sofa, "lounge_sofa", -4.0, 1.0, first, Scale(1.6, 0.8, 0.45)),
                ],
            )
        return world
