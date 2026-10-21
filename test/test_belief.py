from __future__ import annotations

import pytest

from speech2plan.belief import BeliefReport
from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    RoleBinding,
    ThematicRole,
)
from speech2plan.description import EntityDescriber
from speech2plan.grounding import Grounder
from speech2plan.placement import QualifierAttribute
from speech2plan.world_source import PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def report(ward):
    return BeliefReport(
        grounder=Grounder(world=ward), describer=EntityDescriber(world=ward)
    )


def instruction_for(**constraint) -> ParsedInstruction:
    """An instruction whose object role carries one constraint."""
    return ParsedInstruction(
        utterance="take it",
        speaker="person_abc",
        intent=IntentName.GRAB_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT, constraint=EntityConstraint(**constraint)
            )
        ],
        intent_confidence=0.9,
    )


def sentences(report, ward, instruction) -> list:
    return report.sentences(instruction, Grounder(world=ward).ground(instruction))


# %% what matched


def test_candidates_are_reported_with_what_separates_them(report, ward):
    assert sentences(report, ward, instruction_for(entity_type="Cup")) == [
        "I know of 3 cups, 2 of them on this floor: "
        "one in the kitchen on the ground floor "
        "and one in the living room on the ground floor."
    ]


def test_namesakes_in_one_room_are_counted_and_placed(report, ward):
    assert sentences(report, ward, instruction_for(entity_type="Plate")) == [
        "I know of 2 plates, all in the kitchen on the ground floor."
    ]


def test_many_candidates_are_only_counted(ward):
    report = BeliefReport(
        grounder=Grounder(world=ward),
        describer=EntityDescriber(world=ward),
        most_entities=1,
    )
    assert sentences(report, ward, instruction_for(entity_type="Cup")) == [
        "I know of 3 cups, 2 of them on this floor."
    ]


def test_an_assumption_names_what_it_ruled_out(report, ward):
    assert sentences(report, ward, instruction_for(entity_type="Kitchen")) == [
        "There is also the kitchen on the first floor."
    ]


# %% what matched nothing


def test_an_absence_reports_what_the_twin_does_hold(report, ward):
    instruction = instruction_for(
        entity_type="Bathroom",
        qualifiers=[Qualifier(attribute=QualifierAttribute.FLOOR, value="GroundFloor")],
    )
    assert sentences(report, ward, instruction) == [
        "I know of 1 bathroom, the one on the first floor."
    ]


# %% nothing to report


def test_a_resolved_role_adds_nothing(report, ward):
    assert sentences(report, ward, instruction_for(entity_type="Table")) == []
