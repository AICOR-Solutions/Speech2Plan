from __future__ import annotations

import pytest

from speech2plan.grounding import Grounder
from speech2plan.vocabulary import TypeEntry, Vocabulary
from speech2plan.world_source import CareWard, PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def care_ward():
    return CareWard().load()


def entry(vocabulary: Vocabulary, entity_type: str) -> TypeEntry:
    return next(e for e in vocabulary.types if e.entity_type == entity_type)


# %% what the model reads


def test_a_type_held_once_is_listed_alone(ward):
    assert entry(Vocabulary.from_world(ward), "Table").render() == "- Table"


def test_namesakes_are_listed_by_what_sets_them_apart(ward):
    vocabulary = Vocabulary.from_world(ward)
    assert entry(vocabulary, "Kitchen").render() == (
        "- Kitchen (2): floor=GroundFloor | floor=FirstFloor"
    )


def test_what_the_robot_carries_is_not_listed(ward):
    assert "Cup" not in Vocabulary.from_world(ward).entity_types


def test_the_prompt_names_no_instance(care_ward):
    rendered = Vocabulary.from_world(care_ward).render()
    for name in ("kitchen_ground", "bedside_table", "lounge"):
        assert name not in rendered


def test_the_floors_come_from_the_world(ward):
    assert Vocabulary.from_world(ward).floors == ["FirstFloor", "GroundFloor"]


# %% what sets an instance apart really does


def test_every_offered_description_singles_out_its_instance(care_ward):
    grounder = Grounder(world=care_ward)
    vocabulary = Vocabulary.from_world(care_ward)
    for type_entry in vocabulary.types:
        instances = grounder.entities_of_type(type_entry.entity_type)
        for instance, place in zip(instances, type_entry.places):
            matched = grounder.matches(place)
            assert any(match is instance for match in matched)
            shared = [p for p in type_entry.places if p == place]
            assert len(matched) == len(shared), place.notation()


def test_a_type_is_counted(care_ward):
    assert entry(Vocabulary.from_world(care_ward), "Table").count == 2
