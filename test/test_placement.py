from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room

from speech2plan.contract import SpatialRelation
from speech2plan.grounding import Grounder
from speech2plan.placement import (
    PlacementResolver,
    PlacementSnapshot,
    QualifierAttribute,
)
from speech2plan.world_source import CareWard, PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def resolver(ward):
    return PlacementResolver(world=ward)


def named(world, name: str):
    """The place of that name."""
    return next(
        entity
        for entity in Grounder(world=world).searchable_entities()
        if str(entity.name) == name
    )


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


def test_furniture_above_the_floor_is_still_on_that_storey(ward, resolver):
    table = named(ward, "table")
    assert resolver.elevation_of(table) > 0.3
    assert resolver.storey_of(table) == "GroundFloor"


# %% rooms


def test_furniture_resolves_to_the_room_it_stands_in(ward, resolver):
    assert str(resolver.room_containing(named(ward, "table")).name) == "kitchen_ground"


def test_a_place_lies_in_the_room_it_stands_in_only(ward, resolver):
    table = named(ward, "table")
    assert resolver.relates(table, SpatialRelation.IN, named(ward, "kitchen_ground"))
    assert not resolver.relates(table, SpatialRelation.IN, named(ward, "kitchen_first"))
    assert not resolver.relates(
        named(ward, "kitchen_ground"), SpatialRelation.IN, table
    )


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


# %% one moment of the world


def test_a_snapshot_answers_as_the_live_world_does():
    care_ward = CareWard().load()
    live, snapshot = PlacementResolver(world=care_ward), PlacementSnapshot(
        world=care_ward
    )
    for entity in Grounder(world=care_ward).searchable_entities():
        assert snapshot.storey_of(entity) == live.storey_of(entity)
        assert snapshot.room_containing(entity) is live.room_containing(entity)
