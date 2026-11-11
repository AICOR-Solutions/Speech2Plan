from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    Relation,
    RoleBinding,
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


def test_the_room_a_place_is_in_is_a_relation_to_a_room_type(care_ward):
    grounder = Grounder(world=care_ward)
    assert names(
        grounder.matches(
            EntityConstraint(entity_type="SideTable", relations=[inside("Bedroom")])
        )
    ) == ["bedside_table"]


def test_a_reference_is_narrowed_by_its_own_qualifiers(care_ward):
    grounder = Grounder(world=care_ward)
    upstairs = [Qualifier(attribute="floor", value="FirstFloor")]
    assert names(
        grounder.matches(
            EntityConstraint(
                entity_type="Sofa", relations=[inside("LivingRoom", qualifiers=upstairs)]
            )
        )
    ) == ["lounge_sofa"]


def test_two_relations_must_both_hold(care_ward):
    grounder = Grounder(world=care_ward)
    upstairs = [Qualifier(attribute="floor", value="FirstFloor")]
    assert names(
        grounder.matches(
            EntityConstraint(
                entity_type="Table",
                relations=[inside("Kitchen"), inside("Kitchen", qualifiers=upstairs)],
            )
        )
    ) == ["first_floor_table"]
    assert (
        grounder.matches(
            EntityConstraint(
                entity_type="Table", relations=[inside("Kitchen"), inside("Bedroom")]
            )
        )
        == []
    )


def test_a_reference_the_world_lacks_matches_nothing(care_ward):
    outcome = Grounder(world=care_ward).ground_constraint(
        ThematicRole.GOAL,
        EntityConstraint(entity_type="Table", relations=[inside("Garage")]),
    )
    assert isinstance(outcome, Unmatched)


def test_a_place_of_no_known_type_matches_nothing(care_ward):
    outcome = Grounder(world=care_ward).ground_constraint(
        ThematicRole.GOAL,
        EntityConstraint(entity_type=None, spoken_form="the pharmacy"),
    )
    assert isinstance(outcome, Unmatched)


def test_only_rooms_and_furniture_are_places(care_ward):
    grounder = Grounder(world=care_ward)
    assert grounder.entities_of_type("Cup") == []
    assert "towel" not in names(grounder.searchable_entities())
    assert {"kitchen_ground", "bedside_table"} <= set(
        names(grounder.searchable_entities())
    )


def test_what_the_robot_carries_is_not_grounded(care_ward):
    instruction = ParsedInstruction(
        utterance="bring the cup to the bathroom",
        speaker="person_abc",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT,
                constraint=EntityConstraint(entity_type=None, spoken_form="the cup"),
            ),
            RoleBinding(
                role=ThematicRole.GOAL,
                constraint=EntityConstraint(entity_type="Bathroom"),
            ),
        ],
    )

    [outcome] = Grounder(world=care_ward).ground(instruction)

    assert outcome.role == ThematicRole.GOAL
    assert str(outcome.entity.name) == "bathroom"
