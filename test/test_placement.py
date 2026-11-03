from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

from speech2plan.contract import SpatialRelation
from speech2plan.placement import PlacementResolver, QualifierAttribute
from speech2plan.world_source import PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def resolver(ward):
    return PlacementResolver(world=ward)


# %% storeys


def test_every_room_resolves_to_a_storey(ward, resolver):
    assert {
        str(room.name): resolver.storey_of(room)
        for room in ward.get_semantic_annotations_by_type(Room)
    } == {
        "kitchen_ground": "GroundFloor",
        "living_room": "GroundFloor",
        "kitchen_first": "FirstFloor",
        "bathroom": "FirstFloor",
    }


def test_an_object_above_the_floor_is_still_on_that_storey(ward, resolver):
    cup = next(
        entity
        for entity in ward.get_semantic_annotations_by_type(HasGraspCandidates)
        if str(entity.name) == "cup"
    )
    assert resolver.elevation_of(cup) > 0.5
    assert resolver.storey_of(cup) == "GroundFloor"


# %% rooms


def test_objects_resolve_to_the_room_they_stand_in(ward, resolver):
    assert {
        str(entity.name): str(resolver.room_containing(entity).name)
        for entity in ward.get_semantic_annotations_by_type(HasGraspCandidates)
    } == {
        "cup": "kitchen_ground",
        "plate": "kitchen_ground",
        "other_plate": "kitchen_ground",
        "living_room_cup": "living_room",
        "second_cup": "kitchen_first",
    }


# %% qualifiers


def test_a_floor_qualifier_separates_the_two_kitchens(ward, resolver):
    kitchens = [
        room
        for room in ward.get_semantic_annotations_by_type(Room)
        if type(room).__name__ == "Kitchen"
    ]
    on_first = [
        room
        for room in kitchens
        if resolver.satisfies(room, QualifierAttribute.FLOOR, "FirstFloor")
    ]
    assert [str(room.name) for room in on_first] == ["kitchen_first"]


def test_the_room_a_cup_is_in_separates_the_two_cups(ward, resolver):
    kitchen_first = next(
        room
        for room in ward.get_semantic_annotations_by_type(Room)
        if str(room.name) == "kitchen_first"
    )
    cups = [
        entity
        for entity in ward.get_semantic_annotations_by_type(HasGraspCandidates)
        if type(entity).__name__ == "Cup"
    ]
    in_first_kitchen = [
        cup for cup in cups if resolver.relates(cup, SpatialRelation.IN, kitchen_first)
    ]
    assert [str(cup.name) for cup in in_first_kitchen] == ["second_cup"]


# %% what rests on and lies in what


def test_a_snapshot_answers_as_the_live_world_does():
    from speech2plan.placement import PlacementSnapshot
    from speech2plan.world_source import CareWard

    care_ward = CareWard().load()
    live, snapshot = PlacementResolver(world=care_ward), PlacementSnapshot(
        world=care_ward
    )
    for entity in care_ward.get_semantic_annotations_by_type(HasGraspCandidates):
        for resolver_answer in ("support_of", "container_of", "room_containing"):
            assert getattr(snapshot, resolver_answer)(entity) is getattr(
                live, resolver_answer
            )(entity)
