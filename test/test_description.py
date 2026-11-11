from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.world_description.geometry import Scale

from speech2plan.contract import (
    EntityConstraint,
    Qualifier,
    Relation,
    SpatialRelation,
)
from speech2plan.description import EntityDescriber, spoken_words
from speech2plan.grounding import Grounder
from speech2plan.placement import QualifierAttribute
from speech2plan.world_source import CareWard, PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def describer(ward):
    return EntityDescriber(world=ward)


def named(world, name: str):
    """The place of that name."""
    return next(
        entity
        for entity in Grounder(world=world).searchable_entities()
        if str(entity.name) == name
    )


# %% type names become words


def test_a_compound_type_name_becomes_words():
    assert spoken_words("LivingRoom") == "living room"
    assert spoken_words("GroundFloor") == "ground floor"


# %% entities in words


def test_a_room_is_described_by_its_storey(ward, describer):
    assert describer.describe(named(ward, "kitchen_first")).phrase == (
        "the kitchen on the first floor"
    )


def test_furniture_is_described_by_its_room(ward, describer):
    assert describer.describe(named(ward, "table")).phrase == (
        "the table in the kitchen on the ground floor"
    )


def test_the_twins_own_description_wins():
    world = PrototypeWard().load()
    table = named(world, "table")
    with world.modify_world():
        world.add_semantic_annotations(
            [
                NaturalLanguageDescription(
                    root=table.root, description="long oak table by the window"
                )
            ]
        )

    assert EntityDescriber(world=world).describe(table).phrase == (
        "the long oak table by the window"
    )


# %% choices


def test_two_kitchens_are_offered_by_their_floors(ward, describer):
    kitchens = [named(ward, "kitchen_ground"), named(ward, "kitchen_first")]

    choice = describer.choose_between(kitchens)

    assert choice.noun == "kitchen"
    assert choice.count == 2
    assert {
        option: str(entity.name) for option, entity in choice.entity_per_option.items()
    } == {
        "the one on the ground floor": "kitchen_ground",
        "the one on the first floor": "kitchen_first",
    }


def test_namesakes_in_one_room_cannot_be_offered():
    world = PrototypeWard().load()
    with world.modify_world():
        Table.create_with_new_body_in_world(
            name="other_table", world=world, scale=Scale(0.6, 0.6, 0.75)
        )
    tables = [named(world, "table"), named(world, "other_table")]

    choice = EntityDescriber(world=world).choose_between(tables)

    assert not choice.is_separable
    assert choice.noun == "table"
    assert choice.count == 2


# %% from words back to the entity


def test_an_answer_in_words_finds_the_entity(ward, describer):
    kitchens = [named(ward, "kitchen_ground"), named(ward, "kitchen_first")]

    found = describer.entity_for_phrase("on the first floor", kitchens)

    assert str(found.name) == "kitchen_first"


def test_words_nothing_matches_find_nothing(ward, describer):
    assert describer.entity_for_phrase("in the garage", [named(ward, "table")]) is None


# %% constraints in words


def test_a_constraint_is_described_with_its_qualifiers(describer):
    described = describer.describe_constraint(
        EntityConstraint(
            entity_type="Kitchen",
            qualifiers=[
                Qualifier(attribute=QualifierAttribute.FLOOR, value="FirstFloor")
            ],
        )
    )
    assert described == "the kitchen on the first floor"


# %% relations and the care ward


@pytest.fixture(scope="module")
def care_ward():
    return CareWard().load()


def test_a_relation_is_described_in_words(describer):
    described = describer.describe_constraint(
        EntityConstraint(
            entity_type="Table",
            relations=[
                Relation(
                    relation=SpatialRelation.IN,
                    reference=EntityConstraint(entity_type="Kitchen"),
                )
            ],
        )
    )
    assert described == "the table in the kitchen"


def test_a_reference_pinned_by_name_is_described_as_that_entity(care_ward):
    described = EntityDescriber(world=care_ward).describe_constraint(
        EntityConstraint(
            entity_type="SideTable",
            relations=[
                Relation(
                    relation=SpatialRelation.IN,
                    reference=EntityConstraint(
                        entity_type="Bedroom",
                        qualifiers=[
                            Qualifier(
                                attribute=QualifierAttribute.NAME,
                                value="patient_room",
                            )
                        ],
                    ),
                )
            ],
        )
    )
    assert described == "the side table in the bedroom on the ground floor"


def test_a_place_of_no_known_type_is_described_in_the_speakers_words(describer):
    assert (
        describer.describe_constraint(
            EntityConstraint(entity_type=None, spoken_form="the pharmacy")
        )
        == "the pharmacy"
    )


def test_an_answer_saying_where_finds_the_place(care_ward):
    tables = [named(care_ward, "table"), named(care_ward, "first_floor_table")]
    found = EntityDescriber(world=care_ward).entity_for_phrase(
        "in the kitchen on the first floor", tables
    )
    assert str(found.name) == "first_floor_table"
