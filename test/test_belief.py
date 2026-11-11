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

from .fake_preference import NoAssumption

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
    """An instruction whose goal carries one constraint."""
    return ParsedInstruction(
        utterance="go there",
        speaker="person_abc",
        intent=IntentName.MOVE_TO,
        roles=[
            RoleBinding(role=ThematicRole.GOAL, constraint=EntityConstraint(**constraint))
        ],
        intent_confidence=0.9,
    )


def sentences(report, ward, instruction, preference=None) -> list:
    grounder = Grounder(world=ward, preference=preference)
    return report.sentences(instruction, grounder.ground(instruction))


# %% what matched


def test_candidates_are_reported_with_what_separates_them(report, ward):
    assert sentences(
        report, ward, instruction_for(entity_type="Kitchen"), NoAssumption()
    ) == ["I know of 2 kitchens: one on the ground floor and one on the first floor."]


def test_many_candidates_are_only_counted(ward):
    report = BeliefReport(
        grounder=Grounder(world=ward),
        describer=EntityDescriber(world=ward),
        most_entities=1,
    )
    assert sentences(
        report, ward, instruction_for(entity_type="Kitchen"), NoAssumption()
    ) == ["I know of 2 kitchens."]


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


def test_a_kind_of_place_the_world_lacks_is_named_in_the_speakers_words(report, ward):
    instruction = instruction_for(entity_type=None, spoken_form="the pharmacy")
    assert sentences(report, ward, instruction) == ["I know of no pharmacy at all."]
