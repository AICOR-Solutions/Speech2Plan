from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent

from speech2plan.contract import (
    EntityConstraint,
    Qualifier,
    Relation,
    SpatialRelation,
    ThematicRole,
)
from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole, Unmatched
from speech2plan.placement import QualifierResolver

# %% mimics


@dataclass
class MimicFloor:
    """Stands in for a semantic digital twin floor."""


@dataclass
class GroundFloor(MimicFloor): ...


@dataclass
class FirstFloor(MimicFloor): ...


@dataclass
class Kitchen:
    """Stands in for a kitchen annotation, with only what grounding reads."""

    name: str = field(kw_only=True)
    floor: MimicFloor = field(kw_only=True)


@dataclass
class MimicWorld:
    """A world of rooms, with nobody in it."""

    rooms: List[Kitchen] = field(kw_only=True)

    def get_semantic_annotations_by_type(self, semantic_annotation_type) -> List:
        if semantic_annotation_type is Agent:
            return []
        return self.rooms


@dataclass
class FloorTypeResolver(QualifierResolver):
    """Resolves a floor qualifier by the mimic room's floor type."""

    def satisfies(self, entity, attribute: str, value: str) -> bool:
        return attribute == "floor" and type(entity.floor).__name__ == value


# %% fixtures


@pytest.fixture
def two_kitchens() -> MimicWorld:
    return MimicWorld(
        rooms=[
            Kitchen(name="kitchen_ground", floor=GroundFloor()),
            Kitchen(name="kitchen_first", floor=FirstFloor()),
        ]
    )


# %% same-noun disambiguation


def test_two_kitchens_are_ambiguous(two_kitchens):
    grounder = Grounder(world=two_kitchens, qualifier_resolver=FloorTypeResolver())
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL, EntityConstraint(entity_type="Kitchen")
    )
    assert isinstance(outcome, Ambiguity)
    assert outcome.role == ThematicRole.GOAL
    assert len(outcome.candidates) == 2


def test_floor_qualifier_resolves_the_ambiguity(two_kitchens):
    grounder = Grounder(world=two_kitchens, qualifier_resolver=FloorTypeResolver())
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL,
        EntityConstraint(
            entity_type="Kitchen",
            qualifiers=[Qualifier(attribute="floor", value="GroundFloor")],
        ),
    )
    assert isinstance(outcome, ResolvedRole)
    assert outcome.entity.name == "kitchen_ground"


def test_unknown_room_type_does_not_match(two_kitchens):
    grounder = Grounder(world=two_kitchens, qualifier_resolver=FloorTypeResolver())
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL, EntityConstraint(entity_type="Bathroom")
    )
    assert isinstance(outcome, Unmatched)


# %% relations, against the care ward


@pytest.fixture(scope="module")
def care_ward():
    from speech2plan.world_source import CareWard

    return CareWard().load()


def on(entity_type: str, **reference) -> Relation:
    return Relation(
        relation=SpatialRelation.ON,
        reference=EntityConstraint(entity_type=entity_type, **reference),
    )


def inside(entity_type: str, **reference) -> Relation:
    return Relation(
        relation=SpatialRelation.IN,
        reference=EntityConstraint(entity_type=entity_type, **reference),
    )


def names(entities) -> list:
    return [str(entity.name) for entity in entities]


def test_what_a_thing_rests_on_singles_it_out(care_ward):
    grounder = Grounder(world=care_ward)
    assert names(
        grounder.matches(
            EntityConstraint(entity_type="Cup", relations=[on("CounterTop")])
        )
    ) == ["counter_cup"]


def test_the_room_a_thing_is_in_is_a_relation_to_a_room_type(care_ward):
    grounder = Grounder(world=care_ward)
    assert names(
        grounder.matches(
            EntityConstraint(entity_type="Book", relations=[inside("Bedroom")])
        )
    ) == ["patient_book"]


def test_a_reference_is_narrowed_by_its_own_qualifiers(care_ward):
    grounder = Grounder(world=care_ward)
    upstairs = [Qualifier(attribute="floor", value="FirstFloor")]
    assert names(
        grounder.matches(
            EntityConstraint(
                entity_type="Book", relations=[on("Sofa", qualifiers=upstairs)]
            )
        )
    ) == ["lounge_book"]


def test_two_relations_must_both_hold(care_ward):
    grounder = Grounder(world=care_ward)
    assert names(
        grounder.matches(
            EntityConstraint(
                entity_type="Cup", relations=[on("Table"), inside("Kitchen")]
            )
        )
    ) == ["cup", "second_cup"]
    assert (
        grounder.matches(
            EntityConstraint(
                entity_type="Cup", relations=[on("Table"), inside("Bedroom")]
            )
        )
        == []
    )


def test_a_thing_in_a_container_lies_in_it(care_ward):
    grounder = Grounder(world=care_ward)
    assert names(
        grounder.matches(
            EntityConstraint(entity_type="Cloth", relations=[inside("Cabinet")])
        )
    ) == ["towel"]


def test_a_reference_the_world_lacks_matches_nothing(care_ward):
    outcome = Grounder(world=care_ward).ground_constraint(
        ThematicRole.OBJECT,
        EntityConstraint(entity_type="Cup", relations=[on("Fridge")]),
    )
    assert isinstance(outcome, Unmatched)


def test_a_thing_of_no_known_type_matches_nothing(care_ward):
    outcome = Grounder(world=care_ward).ground_constraint(
        ThematicRole.OBJECT,
        EntityConstraint(entity_type=None, spoken_form="the gloves"),
    )
    assert isinstance(outcome, Unmatched)


def test_a_stored_object_rests_where_the_twin_put_it():
    from speech2plan.world_source import CareWard

    world = CareWard().load()
    grounder = Grounder(world=world)
    floating = grounder.entity_named("living_room_cup")
    coffee_table = grounder.entity_named("coffee_table")
    query = EntityConstraint(entity_type="Cup", relations=[on("CoffeeTable")])
    assert grounder.matches(query) == []

    with world.modify_world():
        coffee_table.objects.append(floating)

    assert names(grounder.matches(query)) == ["living_room_cup"]


def test_a_thing_without_a_shape_rests_on_nothing():
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup

    from speech2plan.world_source import CareWard

    world = CareWard().load()
    with world.modify_world():
        Cup.create_with_new_body_in_world(name="perceived_cup", world=world)
    grounder = Grounder(world=world)
    assert "perceived_cup" not in names(
        grounder.matches(EntityConstraint(entity_type="Cup", relations=[on("Table")]))
    )


def test_what_stands_on_a_table_is_not_in_it(care_ward):
    grounder = Grounder(world=care_ward)
    assert (
        grounder.matches(
            EntityConstraint(entity_type="Cup", relations=[inside("SideTable")])
        )
        == []
    )


def test_a_thing_left_unnamed_is_anything_movable_where_the_speaker_said(care_ward):
    outcome = Grounder(world=care_ward).ground_constraint(
        ThematicRole.OBJECT,
        EntityConstraint(entity_type="Thing", relations=[on("CoffeeTable")]),
    )
    assert isinstance(outcome, Ambiguity)
    assert names(outcome.candidates) == ["book", "banana"]


def test_furniture_is_no_unnamed_thing(care_ward):
    found = Grounder(world=care_ward).entities_of_type("Thing")
    assert "coffee_table" not in names(found)
    assert "kitchen_ground" not in names(found)
    assert "towel" in names(found)
