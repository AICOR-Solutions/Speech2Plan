from __future__ import annotations

import json

import pytest
from hri_actions_msgs.msg import Intent

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    Relation,
    RoleBinding,
    RoleConfidence,
    SpatialRelation,
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


# %% the thematic roles


def test_a_resolved_role_carries_the_entity_name(ward):
    message = publish(ward, "take the table", ThematicRole.OBJECT, entity_type="Table")
    assert json.loads(message.data)[ThematicRole.OBJECT] == "table"


def test_an_ambiguous_role_is_not_published_as_a_role(ward):
    message = publish(ward, "take the plate", ThematicRole.OBJECT, entity_type="Plate")
    assert ThematicRole.OBJECT not in json.loads(message.data)


def test_the_utterance_travels_as_input(ward):
    message = publish(ward, "take the table", ThematicRole.OBJECT, entity_type="Table")
    assert json.loads(message.data)[ThematicRole.INPUT] == "take the table"
    assert message.modality == Intent.MODALITY_SPEECH
    assert message.priority == 128


# %% the semantic constraints


def test_a_resolved_constraint_reports_its_entity_and_score(ward):
    message = publish(
        ward,
        "take the cup in the kitchen upstairs",
        ThematicRole.OBJECT,
        entity_type="Cup",
        qualifiers=[Qualifier(attribute=QualifierAttribute.NAME, value="second_cup")],
    )
    assert constraints(message) == [
        {
            "role": ThematicRole.OBJECT,
            "status": GroundingStatus.RESOLVED,
            "confidence": 0.8,
            "entity": "second_cup",
            "assumed": False,
            "entity_type": "Cup",
            "qualifiers": [{"attribute": "name", "value": "second_cup"}],
            "relations": [],
            "spoken_form": None,
            "candidates": [],
        }
    ]


def test_an_ambiguous_constraint_reports_what_matched(ward):
    message = publish(ward, "take the plate", ThematicRole.OBJECT, entity_type="Plate")
    record = constraints(message)[0]
    assert record["status"] == GroundingStatus.AMBIGUOUS
    assert record["entity"] is None
    assert record["candidates"] == ["plate", "other_plate"]


def test_an_assumed_role_is_published_with_what_it_ruled_out(ward):
    message = publish(
        ward, "go to the kitchen", ThematicRole.GOAL, entity_type="Kitchen"
    )
    record = constraints(message)[0]
    assert json.loads(message.data)[ThematicRole.GOAL] == "kitchen_ground"
    assert record["status"] == GroundingStatus.RESOLVED
    assert record["assumed"] is True
    assert record["candidates"] == ["kitchen_first"]


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


def test_a_thing_the_twin_lacks_is_published_where_the_speaker_said_it_is(ward):
    message = publish(
        ward,
        "bring the gloves from the table",
        ThematicRole.OBJECT,
        entity_type=None,
        spoken_form="the gloves",
        relations=[
            Relation(
                relation=SpatialRelation.ON,
                reference=EntityConstraint(
                    entity_type="Table",
                    qualifiers=[
                        Qualifier(attribute=QualifierAttribute.NAME, value="table")
                    ],
                ),
            )
        ],
    )
    record = constraints(message)[0]
    assert ThematicRole.OBJECT not in json.loads(message.data)
    assert record["status"] == GroundingStatus.UNMATCHED
    assert record["entity_type"] is None
    assert record["spoken_form"] == "the gloves"
    assert record["relations"] == [
        {
            "relation": "on",
            "reference": {
                "entity_type": "Table",
                "qualifiers": [{"attribute": "name", "value": "table"}],
                "relations": [],
                "spoken_form": None,
            },
        }
    ]
