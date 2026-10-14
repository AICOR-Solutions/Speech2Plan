from __future__ import annotations

import pytest
from semantic_digital_twin.grasping.grasp_candidates import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

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
        str(entity.name): resolver.room_of(entity)
        for entity in ward.get_semantic_annotations_by_type(HasGraspCandidates)
    } == {
        "cup": "kitchen_ground",
        "plate": "kitchen_ground",
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


def test_a_room_qualifier_separates_the_two_cups(ward, resolver):
    cups = [
        entity
        for entity in ward.get_semantic_annotations_by_type(HasGraspCandidates)
        if type(entity).__name__ == "Cup"
    ]
    in_first_kitchen = [
        cup
        for cup in cups
        if resolver.satisfies(cup, QualifierAttribute.ROOM, "kitchen_first")
    ]
    assert [str(cup.name) for cup in in_first_kitchen] == ["second_cup"]
