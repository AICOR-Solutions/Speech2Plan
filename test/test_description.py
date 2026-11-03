from __future__ import annotations

import pytest
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Furniture,
    Room,
)

from speech2plan.contract import (
    EntityConstraint,
    Qualifier,
    Relation,
    SpatialRelation,
)
from speech2plan.description import EntityDescriber, spoken_words
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
    """The entity of that name, room or object."""
    for annotation_type in (Room, HasGraspCandidates, Furniture):
        for entity in world.get_semantic_annotations_by_type(annotation_type):
            if str(entity.name) == name:
                return entity
    raise LookupError(name)


# %% type names become words


def test_a_compound_type_name_becomes_words():
    assert spoken_words("LivingRoom") == "living room"
    assert spoken_words("GroundFloor") == "ground floor"


# %% entities in words


def test_a_room_is_described_by_its_storey(ward, describer):
    assert describer.describe(named(ward, "kitchen_first")).phrase == (
        "the kitchen on the first floor"
    )


def test_an_object_is_described_by_its_room(ward, describer):
    assert describer.describe(named(ward, "cup")).phrase == (
        "the cup in the kitchen on the ground floor"
    )


def test_the_twins_own_description_wins(ward):
    world = PrototypeWard().load()
    cup = named(world, "cup")
    with world.modify_world():
        world.add_semantic_annotations(
            [
                NaturalLanguageDescription(
                    root=cup.root, description="blue mug with a chipped handle"
                )
            ]
        )

    assert EntityDescriber(world=world).describe(cup).phrase == (
        "the blue mug with a chipped handle"
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


def test_namesakes_in_one_room_cannot_be_offered(ward, describer):
    plates = [named(ward, "plate"), named(ward, "other_plate")]

    choice = describer.choose_between(plates)

    assert not choice.is_separable
    assert choice.noun == "plate"
    assert choice.count == 2


# %% from words back to the entity


def test_an_answer_in_words_finds_the_entity(ward, describer):
    cups = [named(ward, "cup"), named(ward, "second_cup")]

    found = describer.entity_for_phrase("in the kitchen on the first floor", cups)

    assert str(found.name) == "second_cup"


def test_words_nothing_matches_find_nothing(ward, describer):
    assert describer.entity_for_phrase("in the garage", [named(ward, "cup")]) is None


# %% constraints in words


def test_a_constraint_is_described_with_its_qualifiers(describer):
    described = describer.describe_constraint(
        EntityConstraint(
            entity_type="Cup",
            qualifiers=[
                Qualifier(attribute=QualifierAttribute.FLOOR, value="FirstFloor")
            ],
        )
    )
    assert described == "the cup on the first floor"


# %% relations and the care ward


@pytest.fixture(scope="module")
def care_ward():
    return CareWard().load()


def test_a_relation_is_described_in_words(describer):
    described = describer.describe_constraint(
        EntityConstraint(
            entity_type="Cup",
            relations=[
                Relation(
                    relation=SpatialRelation.ON,
                    reference=EntityConstraint(entity_type="Table"),
                )
            ],
        )
    )
    assert described == "the cup on the table"


def test_a_reference_pinned_by_name_is_described_as_that_entity(care_ward):
    described = EntityDescriber(world=care_ward).describe_constraint(
        EntityConstraint(
            entity_type="Cup",
            relations=[
                Relation(
                    relation=SpatialRelation.ON,
                    reference=EntityConstraint(
                        entity_type="SideTable",
                        qualifiers=[
                            Qualifier(
                                attribute=QualifierAttribute.NAME,
                                value="bedside_table",
                            )
                        ],
                    ),
                )
            ],
        )
    )
    assert described == ("the cup on the side table in the bedroom on the ground floor")


def test_a_thing_of_no_known_type_is_described_in_the_speakers_words(describer):
    assert (
        describer.describe_constraint(
            EntityConstraint(entity_type=None, spoken_form="the gloves")
        )
        == "the gloves"
    )


def test_what_a_thing_stands_on_sets_apart_namesakes_in_one_room(care_ward):
    cups = [named(care_ward, "cup"), named(care_ward, "counter_cup")]
    choice = EntityDescriber(world=care_ward).choose_between(cups)
    assert {
        option: str(entity.name) for option, entity in choice.entity_per_option.items()
    } == {
        "the one on the table": "cup",
        "the one on the counter top": "counter_cup",
    }


def test_what_a_thing_lies_in_is_a_feature(care_ward):
    describer = EntityDescriber(world=care_ward)
    assert describer.place_feature(named(care_ward, "towel")) == "in the cabinet"


def test_an_answer_saying_where_finds_the_entity(care_ward):
    cups = [named(care_ward, "cup"), named(care_ward, "counter_cup")]
    found = EntityDescriber(world=care_ward).entity_for_phrase(
        "on the counter top", cups
    )
    assert str(found.name) == "counter_cup"
