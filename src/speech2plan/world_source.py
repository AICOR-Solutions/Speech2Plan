from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type

import numpy as np
import urdf_parser_py.xml_reflection.core as urdf_reflection
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.building_floor import BuildingFloor
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    Bedroom,
    Cup,
    Door,
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Human,
    Kitchen,
    LivingRoom,
    Plate,
    Room,
    Table,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% interface


class WorldSource(ABC):
    """Where the world the parser grounds against comes from."""

    @abstractmethod
    def load(self) -> World:
        """Return a world."""

    def is_available(self) -> bool:
        """True when `load` can succeed now."""
        return True


# %% worlds built in process


@dataclass
class BuiltWorld(WorldSource, ABC):
    """A world built in this process, two storeys high."""

    storey_height: float = field(kw_only=True, default=3.0)
    footprint: float = field(kw_only=True, default=20.0)
    """The side length of the square each storey spans."""

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


# %% prototyping


@dataclass
class PrototypeWard(BuiltWorld):
    """A two storey building with a kitchen on each floor and a few objects.

    The nurse stands on the ground floor, which is what makes "the kitchen"
    resolve without asking. Two of the three cups are on her floor, in different
    rooms, so words tell those apart. The two plates share one kitchen, which is
    the case words cannot settle.
    """

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
            first = self.storey_height
            for annotation_type, name, x, z in (
                (Table, "table", 1.0, 0.0),
                (Cup, "cup", 1.0, 0.8),
                (Plate, "plate", 1.2, 0.8),
                (Plate, "other_plate", 1.4, 0.8),
                (Cup, "living_room_cup", 5.5, 0.8),
                (Cup, "second_cup", 1.0, first + 0.8),
                (Human, "nurse", 0.5, 0.0),
            ):
                self.add_box(
                    world,
                    annotation_type,
                    name,
                    Point3(x, 1.0, z),
                    Scale(0.1, 0.1, 0.1),
                )
        return world

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


# %% a building with an elevator


class Corner(Enum):
    """Where `BuildingFloor` puts its rooms, by the index it names them with."""

    NORTH_WEST = 0
    SOUTH_WEST = 1
    NORTH_EAST = 2
    SOUTH_EAST = 3


@dataclass
class ElevatorBuilding(BuiltWorld):
    """The two storey building of the CRAM elevator tests, with named rooms.

    Each storey is a `BuildingFloor`, whose four corner rooms carry no Room
    annotation of their own, so every room is given a type and a name here. The
    ground floor has a kitchen with a cup on its table, a day room with an empty
    table, a bathroom and a patient room. The first floor repeats the kitchen and
    the bathroom, so "the kitchen" needs the floor, and has a bedroom with a
    table. The elevator connects both storeys; the robot and the nurse stand in
    the ground floor corridor.
    """

    footprint: float = field(kw_only=True, default=8.0)
    robot_type: Optional[Type[AbstractRobot]] = field(kw_only=True, default=Tiago)
    """The robot standing in the corridor, or None for a building without one."""

    def load(self) -> World:
        world = World.create_with_root_body("root")
        BuildingFloor().spawn(world, "floor_1")
        BuildingFloor().spawn(
            world,
            "floor_2",
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=self.storey_height
            ),
        )
        self.add_elevator(world)
        with world.modify_world():
            self.add_storeys(world)
            self.name_rooms(world)
            self.add_furniture(world)
        if self.robot_type is not None:
            self.add_robot(world)
        return world

    def name_rooms(self, world: World) -> None:
        """Annotate the corner rooms of both storeys as typed, named rooms."""
        rooms = (
            ("floor_1", Corner.NORTH_WEST, Kitchen, "kitchen_ground"),
            ("floor_1", Corner.SOUTH_WEST, Bathroom, "bathroom_ground"),
            ("floor_1", Corner.NORTH_EAST, LivingRoom, "day_room"),
            ("floor_1", Corner.SOUTH_EAST, Bedroom, "patient_room"),
            ("floor_2", Corner.NORTH_WEST, Kitchen, "kitchen_first"),
            ("floor_2", Corner.SOUTH_WEST, Bathroom, "bathroom_first"),
            ("floor_2", Corner.NORTH_EAST, Bedroom, "bedroom_first"),
            ("floor_2", Corner.SOUTH_EAST, LivingRoom, "lounge"),
        )
        world.add_semantic_annotations(
            [
                room_type(
                    floor=world.get_semantic_annotation_by_name(
                        f"{storey}_room_{corner.value}_floor"
                    ),
                    name=PrefixedName(name),
                )
                for storey, corner, room_type, name in rooms
            ]
        )

    def add_furniture(self, world: World) -> None:
        """Add the tables, against the back wall of their rooms, the cups and the nurse."""
        first = self.storey_height
        for name, x, elevation in (
            ("kitchen_table", -1.0, 0.0),
            ("day_room_table", 1.0, 0.0),
            ("kitchen_first_table", -1.0, first),
            ("bedroom_first_table", 1.0, first),
        ):
            self.add_box(
                world,
                Table,
                name,
                Point3(x, 3.1, elevation + 0.375),
                Scale(0.5, 0.5, 0.75),
            )
        cup = Scale(0.06, 0.06, 0.1)
        self.add_box(world, Cup, "cup", Point3(-1.0, 3.0, 0.8), cup)
        self.add_box(world, Cup, "first_floor_cup", Point3(-1.0, 3.0, first + 0.8), cup)
        self.add_box(world, Human, "nurse", Point3(2.5, 0.0, 0.0), Scale(0.3, 0.3, 0.1))

    def add_elevator(self, world: World) -> None:
        """Add the elevator of the CRAM tests next to the building, opening onto it."""
        cabin_scale = Scale(2, 2, 2)
        door_scale = Scale(0.05, cabin_scale.y / 2, cabin_scale.z)
        shaft = World()
        with shaft.modify_world():
            shaft.add_body(Body(name=PrefixedName("elevator_shaft")))
            elevator = Elevator.get_annotation_specification(
                "elevator",
                Elevator.get_default_root_kinematic_structure_entity_specification(
                    scale=cabin_scale, wall_thickness=0.05
                ),
                parent_connection_specification=PrismaticConnectionSpecification(
                    axis=Vector3.Z(),
                    dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                        maximum_speed=1.0
                    ),
                ),
            ).spawn(shaft)
            for index, door_y in enumerate((-cabin_scale.y / 4, cabin_scale.y / 4)):
                elevator.add(
                    Door.create_with_new_body_in_world(
                        name=f"elevator_door{index}",
                        world=shaft,
                        world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            Point3(-cabin_scale.x / 2, door_y, 0),
                            reference_frame=shaft.root,
                        ),
                        parent_connection_specification=PrismaticConnectionSpecification(
                            axis=Vector3.Y() * ((-1) ** (index + 1)),
                            dof_limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                                lower_position=0.0, upper_position=door_scale.y
                            ),
                        ),
                        scale=door_scale,
                    )
                )
            shaft.add_semantic_annotation(elevator)
        world.merge_world(
            shaft,
            FixedConnection(
                parent=world.root,
                child=shaft.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    -5, 0, 1, yaw=np.pi
                ),
            ),
        )
        elevator.open()

    def add_robot(self, world: World) -> None:
        """Parse the robot's URDF, without its parser warnings, and stand it in the corridor."""
        report = urdf_reflection.on_error
        urdf_reflection.on_error = lambda message: None
        try:
            robot_world = URDFParser.from_file(
                file_path=self.robot_type.get_ros_file_path(),
                use_visual_as_collision_backup=self.robot_type.uses_visual_as_collision_backup,
            ).parse()
        finally:
            urdf_reflection.on_error = report
        self.robot_type.from_world(robot_world)
        with robot_world.modify_world():
            robot_root = robot_world.root
            odometry = Body(name=PrefixedName("odom_combined"))
            robot_world.add_connection(
                Connection6DoF.create_with_dofs(
                    robot_world, Body(name=PrefixedName("map")), odometry
                )
            )
            drive = self.robot_type.get_drive_connection_type().create_with_dofs(
                parent=odometry, child=robot_root, world=robot_world
            )
            robot_world.add_connection(drive)
            drive.has_hardware_interface = True
        world.merge_world(robot_world)
        robot = world.get_semantic_annotations_by_type(self.robot_type)[0]
        robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                1, 0, 0, reference_frame=world.root
            )
        )
