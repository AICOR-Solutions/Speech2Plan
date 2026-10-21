from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.failures import CandidateLimitReached
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
)
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from hri_actions_msgs.msg import Intent
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    FirstFloor,
)

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.exceptions import (
    IntentNotPlannable,
    NoSurfaceInRoom,
    RoleNotGrounded,
)
from speech2plan.grounding import Grounder
from speech2plan.intent_message import PublishedIntent
from speech2plan.planning import KinematicSimulation, PlanBuilder, actions_of
from speech2plan.world_source import ElevatorBuilding

# %% fixtures


@pytest.fixture(scope="module")
def building():
    return ElevatorBuilding().load()


@pytest.fixture
def context(building) -> Context:
    world = deepcopy(building)
    return Context(
        world,
        world.get_semantic_annotations_by_type(Tiago)[0],
        alternative_motion_mappings=[TiagoMoveSim],
    )


def tasks(steps) -> list:
    """The action types of the steps, without the drives and the arm parking."""
    return [
        step._type_ if isinstance(step, Match) else type(step)
        for step in actions_of(steps)
        if not isinstance(step, (ParkArmsAction, NavigateAction))
    ]


def drive_targets(steps) -> list:
    """Where the drives among the steps end, as x, y and z."""
    return [
        step.target_location.position.to_np().flatten()[:3].tolist()
        for step in actions_of(steps)
        if isinstance(step, NavigateAction)
    ]


def position_of(entity) -> np.ndarray:
    return entity.root.global_pose.position.to_np().flatten()[:3]


# %% which actions an intent becomes


def test_moving_to_a_room_on_the_robots_floor_is_one_drive_to_its_middle(context):
    builder = PlanBuilder(context=context)
    room = context.world.get_semantic_annotation_by_name("day_room")

    steps = builder.steps_for(IntentName.MOVE_TO, {ThematicRole.GOAL: "day_room"})

    assert tasks(steps) == []
    assert drive_targets(steps) == [
        builder.middle_of(room).position.to_np().flatten()[:3].tolist()
    ]


def test_moving_to_another_floor_rides_the_elevator_first(context):
    steps = PlanBuilder(context=context).steps_for(
        IntentName.MOVE_TO, {ThematicRole.GOAL: "bedroom_first"}
    )
    assert tasks(steps) == [ElevatorNavigation]
    assert isinstance(steps[0].actions[0].target_floor, FirstFloor)


def test_bringing_on_one_floor_picks_and_places(context):
    steps = PlanBuilder(context=context).steps_for(
        IntentName.BRING_OBJECT,
        {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "day_room"},
    )
    assert tasks(steps) == [MoveAndPickUpAction, MoveAndPlaceAction]


def test_bringing_upstairs_rides_between_picking_and_placing(context):
    steps = PlanBuilder(context=context).steps_for(
        IntentName.BRING_OBJECT,
        {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "bedroom_first"},
    )
    assert tasks(steps) == [MoveAndPickUpAction, ElevatorNavigation, MoveAndPlaceAction]


def test_fetching_from_upstairs_rides_before_picking(context):
    steps = PlanBuilder(context=context).steps_for(
        IntentName.GRAB_OBJECT, {ThematicRole.OBJECT: "first_floor_cup"}
    )
    assert tasks(steps) == [ElevatorNavigation, MoveAndPickUpAction]


# %% what the steps are called


def test_each_step_is_named_in_words(context):
    steps = PlanBuilder(context=context).steps_for(
        IntentName.BRING_OBJECT,
        {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "bedroom_first"},
    )
    assert [step.summary for step in steps] == [
        "pick up cup in kitchen_ground",
        "ride the elevator to the first floor",
        "put cup on bedroom_first_table",
    ]


# %% what cannot be planned


def test_an_intent_without_a_plan_is_refused(context):
    with pytest.raises(IntentNotPlannable):
        PlanBuilder(context=context).steps_for(IntentName.GUIDE, {})


def test_a_role_the_intent_left_open_is_refused(context):
    with pytest.raises(RoleNotGrounded) as raised:
        PlanBuilder(context=context).steps_for(
            IntentName.BRING_OBJECT, {ThematicRole.GOAL: "day_room"}
        )
    assert raised.value.role == ThematicRole.OBJECT


def test_a_room_with_nothing_to_put_things_on_is_refused(context):
    with pytest.raises(NoSurfaceInRoom):
        PlanBuilder(context=context).steps_for(
            IntentName.BRING_OBJECT,
            {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "bathroom_ground"},
        )


# %% from the published message


def test_the_published_intent_is_planned_from_its_message(building, context):
    instruction = ParsedInstruction(
        utterance="go to the bedroom",
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.MOVE_TO,
        roles=[
            RoleBinding(
                role=ThematicRole.GOAL,
                constraint=EntityConstraint(entity_type="Bedroom"),
            )
        ],
        intent_confidence=0.9,
        confidence_per_role=[RoleConfidence(role=ThematicRole.GOAL, confidence=0.9)],
    )
    message = PublishedIntent(
        instruction=instruction, outcomes=Grounder(world=building).ground(instruction)
    ).as_message()

    assert json.loads(message.data)[ThematicRole.GOAL] == "patient_room"
    builder = PlanBuilder(context=context)
    patient_room = context.world.get_semantic_annotation_by_name("patient_room")
    steps = builder.steps_for(message.intent, json.loads(message.data))
    assert drive_targets(steps) == [
        builder.middle_of(patient_room).position.to_np().flatten()[:3].tolist()
    ]


# %% carrying the plans out


def test_the_robot_rides_up_and_drives_into_the_room(context):
    builder = PlanBuilder(context=context)
    room = context.world.get_semantic_annotation_by_name("bedroom_first")
    target = builder.middle_of(room).position.to_np().flatten()[:3]

    KinematicSimulation(context=context).perform(
        builder.steps_for(IntentName.MOVE_TO, {ThematicRole.GOAL: "bedroom_first"}),
    )

    assert position_of(context.robot) == pytest.approx(target, abs=0.01)


def test_the_cup_ends_up_on_the_table_of_the_room(context):
    builder = PlanBuilder(context=context)
    cup = context.world.get_semantic_annotation_by_name("cup")
    table = context.world.get_semantic_annotation_by_name("day_room_table")
    target = builder.placement_on(table, cup).position.to_np().flatten()[:3]

    KinematicSimulation(context=context).perform(
        builder.steps_for(
            IntentName.BRING_OBJECT,
            {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "day_room"},
        ),
    )

    assert position_of(cup) == pytest.approx(target, abs=0.02)


@pytest.mark.xfail(
    raises=CandidateLimitReached,
    strict=True,
    reason="coraplex samples standing poses as if every storey were the ground floor",
)
def test_the_cup_is_carried_upstairs(context):
    builder = PlanBuilder(context=context)
    cup = context.world.get_semantic_annotation_by_name("cup")
    table = context.world.get_semantic_annotation_by_name("bedroom_first_table")
    target = builder.placement_on(table, cup).position.to_np().flatten()[:3]

    KinematicSimulation(context=context).perform(
        builder.steps_for(
            IntentName.BRING_OBJECT,
            {ThematicRole.OBJECT: "cup", ThematicRole.GOAL: "bedroom_first"},
        ),
    )

    assert position_of(cup) == pytest.approx(target, abs=0.02)
