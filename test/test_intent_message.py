from __future__ import annotations

import json

import pytest
from hri_actions_msgs.msg import Intent

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.grounding import Grounder
from speech2plan.intent_message import GroundingStatus, PayloadKey, PublishedIntent
from speech2plan.placement import QualifierAttribute
from speech2plan.world_source import PrototypeWard

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


def publish(ward, utterance: str, role: ThematicRole, **constraint) -> Intent:
    """The Intent the node would publish for one parsed role."""
    instruction = ParsedInstruction(
        utterance=utterance,
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.BRING_OBJECT,
        roles=[RoleBinding(role=role, constraint=EntityConstraint(**constraint))],
        intent_confidence=0.9,
        confidence_per_role=[RoleConfidence(role=role, confidence=0.8)],
    )
    grounder = Grounder(world=ward)
    return PublishedIntent(
        instruction=instruction, outcomes=grounder.ground(instruction)
    ).as_message()


def constraints(message: Intent) -> list:
    return json.loads(message.data)[PayloadKey.OTHER_DATA][PayloadKey.CONSTRAINTS]


def item_message(spoken_form: str) -> Intent:
    """The Intent for bringing what the speaker called `spoken_form`."""
    instruction = ParsedInstruction(
        utterance=f"bring {spoken_form}",
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT,
                constraint=EntityConstraint(entity_type=None, spoken_form=spoken_form),
            )
        ],
        intent_confidence=0.9,
    )
    return PublishedIntent(instruction=instruction, outcomes=[]).as_message()


# %% the thematic roles


def test_a_resolved_place_carries_the_entity_name(ward):
    message = publish(ward, "go to the table", ThematicRole.GOAL, entity_type="Table")
    assert json.loads(message.data)[ThematicRole.GOAL] == "table"


def test_a_place_that_matched_nothing_is_not_sent_as_a_role(ward):
    message = publish(ward, "go to the garage", ThematicRole.GOAL, entity_type="Garage")
    assert ThematicRole.GOAL not in json.loads(message.data)


def test_the_item_is_sent_as_the_speakers_word_for_it():
    assert json.loads(item_message("my pills").data)[ThematicRole.OBJECT] == "pills"
    assert json.loads(item_message("the insulin").data)[ThematicRole.OBJECT] == (
        "insulin"
    )


def test_the_item_is_no_constraint():
    assert constraints(item_message("the insulin")) == []


def test_the_utterance_travels_as_input(ward):
    message = publish(ward, "go to the table", ThematicRole.GOAL, entity_type="Table")
    assert json.loads(message.data)[PayloadKey.INPUT] == "go to the table"
    assert message.modality == Intent.MODALITY_SPEECH
    assert message.priority == 128


# %% the semantic constraints


def test_a_resolved_constraint_reports_its_entity_and_score(ward):
    message = publish(
        ward,
        "go to the kitchen upstairs",
        ThematicRole.GOAL,
        entity_type="Kitchen",
        qualifiers=[
            Qualifier(attribute=QualifierAttribute.NAME, value="kitchen_first")
        ],
    )
    assert constraints(message) == [
        {
            "role": ThematicRole.GOAL,
            "status": GroundingStatus.RESOLVED,
            "confidence": 0.8,
            "entity": "kitchen_first",
            "assumed": False,
            "entity_type": "Kitchen",
            "qualifiers": [{"attribute": "name", "value": "kitchen_first"}],
            "relations": [],
            "spoken_form": None,
            "candidates": [],
        }
    ]


def test_an_assumed_role_is_published_with_what_it_ruled_out(ward):
    message = publish(
        ward, "go to the kitchen", ThematicRole.GOAL, entity_type="Kitchen"
    )
    record = constraints(message)[0]
    assert json.loads(message.data)[ThematicRole.GOAL] == "kitchen_ground"
    assert record["status"] == GroundingStatus.RESOLVED
    assert record["assumed"] is True
    assert record["candidates"] == ["kitchen_first"]


def test_an_unmatched_place_reports_the_speakers_words(ward):
    message = publish(
        ward,
        "go to the pharmacy",
        ThematicRole.GOAL,
        entity_type=None,
        spoken_form="the pharmacy",
    )
    record = constraints(message)[0]
    assert record["status"] == GroundingStatus.UNMATCHED
    assert record["spoken_form"] == "the pharmacy"


def test_an_unfilled_role_is_reported_without_a_type(ward):
    instruction = ParsedInstruction(
        utterance="bring this here",
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.BRING_OBJECT,
        unfilled_roles=[ThematicRole.OBJECT],
        intent_confidence=0.9,
    )
    message = PublishedIntent(instruction=instruction, outcomes=[]).as_message()

    assert ThematicRole.OBJECT not in json.loads(message.data)
    assert constraints(message)[0]["status"] == GroundingStatus.UNFILLED
    assert constraints(message)[0]["entity_type"] is None
