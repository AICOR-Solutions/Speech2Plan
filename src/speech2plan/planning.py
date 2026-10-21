from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Floor,
    Level,
    Room,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import IntentName, ThematicRole
from speech2plan.description import spoken_words
from speech2plan.exceptions import (
    IntentNotPlannable,
    NoElevator,
    NoSurfaceInRoom,
    RoleNotGrounded,
)
from speech2plan.placement import PlacementResolver

# %% intent to plan


@dataclass
class PlanStep:
    """One step of a plan as a person would name it, with the actions doing it."""

    summary: str = field(kw_only=True)
    actions: List[ActionDescription] = field(kw_only=True)


@dataclass
class PlanBuilder:
    """Builds a coraplex plan from an hri_actions_msgs/Intent.

    Reads only the thematic roles of `data`, which name world entities, so it
    plans any conformant intent, not only those Speech2Plan publishes.
    """

    context: Context = field(kw_only=True)

    def __post_init__(self):
        self.resolver = PlacementResolver(world=self.context.world)

    def steps_for(self, intent: str, roles: dict) -> List[PlanStep]:
        """The steps carrying out an intent, in order."""
        builders: Dict[str, Callable[[str, dict], List[PlanStep]]] = {
            IntentName.MOVE_TO: self.move_to,
            IntentName.GRAB_OBJECT: self.grab,
            IntentName.BRING_OBJECT: self.bring,
        }
        if intent not in builders:
            raise IntentNotPlannable(
                intent=intent, plannable=[str(name) for name in builders]
            )
        return builders[intent](intent, roles)

    def move_to(self, intent: str, roles: dict) -> List[PlanStep]:
        goal = self.entity(intent, roles, ThematicRole.GOAL)
        return [
            *self.ride(
                self.resolver.level_of(self.robot), self.resolver.level_of(goal)
            ),
            PlanStep(
                summary=f"drive to {goal.name}",
                actions=[NavigateAction(self.middle_of(goal))],
            ),
        ]

    def grab(self, intent: str, roles: dict) -> List[PlanStep]:
        item = self.entity(intent, roles, ThematicRole.OBJECT)
        return [
            *self.ride(
                self.resolver.level_of(self.robot), self.resolver.level_of(item)
            ),
            self.picking_up(
                item,
                MoveAndPickUpAction.from_graspable_by_closest_grasps(
                    graspable=item, arm=self.arm, context=self.context
                ),
            ),
        ]

    def bring(self, intent: str, roles: dict) -> List[PlanStep]:
        item = self.entity(intent, roles, ThematicRole.OBJECT)
        goal = self.entity(intent, roles, ThematicRole.GOAL)
        surface = self.surface_for(goal)
        transport = TransportAction.from_graspable_by_closest_grasps(
            item, self.placement_on(surface, item), self.arm, self.context
        )
        return [
            *self.ride(
                self.resolver.level_of(self.robot), self.resolver.level_of(item)
            ),
            self.picking_up(item, transport.pick_up),
            *self.ride(self.resolver.level_of(item), self.resolver.level_of(goal)),
            PlanStep(
                summary=f"put {item.name} on {surface.name}",
                actions=[transport.place, ParkArmsAction(self.robot.all_arms)],
            ),
        ]

    def picking_up(self, item: SemanticAnnotation, pick_up) -> PlanStep:
        """The pick up of `item`, with the arms parked before and after."""
        return PlanStep(
            summary=f"pick up {item.name} in {self.resolver.room_of(item)}",
            actions=[
                ParkArmsAction(self.robot.all_arms),
                pick_up,
                ParkArmsAction(self.robot.all_arms),
            ],
        )

    # %% reading the intent

    def entity(
        self, intent: str, roles: dict, role: ThematicRole
    ) -> SemanticAnnotation:
        """The world entity a role names."""
        if roles.get(str(role)) is None:
            raise RoleNotGrounded(role=str(role), intent=intent)
        return self.context.world.get_semantic_annotation_by_name(roles[str(role)])

    # %% where things are

    @property
    def robot(self) -> AbstractRobot:
        return self.context.robot

    @property
    def arm(self) -> Arm:
        return self.robot.get_right_arm_if_specified() or self.robot.all_arms[0]

    def ride(self, start: Optional[Level], end: Optional[Level]) -> List[PlanStep]:
        """The elevator ride from one storey to another, if they differ."""
        if start is end:
            return []
        elevators = self.context.world.get_semantic_annotations_by_type(Elevator)
        if not elevators:
            raise NoElevator(level=str(end.name))
        return [
            PlanStep(
                summary=f"ride the elevator to the {spoken_words(type(end).__name__)}",
                actions=[ElevatorNavigation(elevators[0], end)],
            )
        ]

    def middle_of(self, entity: SemanticAnnotation) -> Pose:
        """The middle of the entity's footprint, on the floor it stands on."""
        extent = self.extent_of(entity)
        return Pose.from_xyz_rpy(
            (extent.min_x + extent.max_x) / 2,
            (extent.min_y + extent.max_y) / 2,
            float(self.resolver.pose_of(entity).position.z),
            reference_frame=self.context.world.root,
        )

    def extent_of(self, entity: SemanticAnnotation) -> VolumetricBoundingBox:
        """The entity's bounding box in the world root frame, walls of a room included."""
        placed = entity.floor if isinstance(entity, Room) else entity
        return placed.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=self.context.world.root)
        ).bounding_box()

    def surface_for(self, goal: SemanticAnnotation) -> SemanticAnnotation:
        """Where to put something down: the goal, or a surface in the goal room."""
        if not isinstance(goal, Room):
            return goal
        surfaces = [
            surface
            for surface in self.context.world.get_semantic_annotations_by_type(
                HasSupportingSurface
            )
            if not isinstance(surface, Floor)
            and self.resolver.room_containing(surface) is goal
        ]
        if not surfaces:
            raise NoSurfaceInRoom(room=str(goal.name))
        return surfaces[0]

    def placement_on(
        self, surface: SemanticAnnotation, item: SemanticAnnotation
    ) -> Pose:
        """Where `item` stands once put down in the middle of `surface`."""
        top = self.extent_of(surface)
        height = self.extent_of(item).max_z - self.extent_of(item).min_z
        return Pose.from_xyz_rpy(
            (top.min_x + top.max_x) / 2,
            (top.min_y + top.max_y) / 2,
            top.max_z + height / 2,
            reference_frame=self.context.world.root,
        )


def actions_of(steps: List[PlanStep]) -> List[ActionDescription]:
    """The actions of `steps`, in order."""
    return [action for step in steps for action in step.actions]


# %% the elevator in simulation


@dataclass(eq=False)
class SimulatedElevatorOperator(ModelChangeCallback):
    """Sends the elevator to the next storey of a plan whenever the robot boards it.

    Stands in for whoever operates the elevator when the plan runs in simulation.
    """

    elevator: Elevator = field(kw_only=True)
    robot: AbstractRobot = field(kw_only=True)
    destinations: List[Level] = field(kw_only=True)
    """The storeys the plan rides to, in order."""

    aboard: bool = field(init=False, default=False)

    @classmethod
    def for_actions(
        cls, actions: List[ActionDescription], context: Context
    ) -> Optional[SimulatedElevatorOperator]:
        """An operator for the rides among `actions`, or None when there are none."""
        rides = [action for action in actions if isinstance(action, ElevatorNavigation)]
        if not rides:
            return None
        return cls(
            _world=context.world,
            elevator=rides[0].elevator,
            robot=context.robot,
            destinations=[ride.target_floor for ride in rides],
        )

    def on_model_change(self, **kwargs):
        boarded = (
            self.robot.root.parent_kinematic_structure_entity is self.elevator.root
        )
        if boarded and not self.aboard and self.destinations:
            self.elevator.close()
            self.elevator.drive_to_floor(self.destinations.pop(0))
            self.elevator.open()
        self.aboard = boarded


@dataclass
class KinematicSimulation:
    """Carries out a plan on the kinematically simulated robot of a context."""

    context: Context = field(kw_only=True)

    def perform(self, steps: List[PlanStep]) -> None:
        """Run `steps`, with the elevator operated for the rides among them."""
        actions = actions_of(steps)
        plan = sequential(actions, context=self.context)
        operator = SimulatedElevatorOperator.for_actions(actions, self.context)
        try:
            with kinematically_simulated_robot:
                plan.perform()
        finally:
            if operator is not None:
                operator.stop()
