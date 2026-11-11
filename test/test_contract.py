from __future__ import annotations

from hri_actions_msgs.msg import Intent

from speech2plan.contract import (
    IntentName,
    ParsedInstruction,
    RoleConfidence,
    ThematicRole,
)

# %% the vocabulary is the message's, not a copy


def test_intent_names_match_the_ros4hri_message():
    assert IntentName.BRING_OBJECT == Intent.BRING_OBJECT
    assert IntentName.MOVE_TO == Intent.MOVE_TO
    assert IntentName.RAW_USER_INPUT == Intent.RAW_USER_INPUT


def test_only_the_intents_of_the_transport_mission_are_recognised():
    assert [str(intent) for intent in IntentName] == [
        Intent.RAW_USER_INPUT,
        Intent.MOVE_TO,
        Intent.BRING_OBJECT,
    ]


def test_thematic_roles_are_the_documented_keys():
    assert ThematicRole.OBJECT == "object"
    assert ThematicRole.GOAL == "goal"
    assert ThematicRole.RECIPIENT == "recipient"


# %% confidence


def test_the_overall_confidence_is_the_weakest_part():
    instruction = ParsedInstruction(
        utterance="bring the cup to the kitchen",
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.BRING_OBJECT,
        intent_confidence=0.95,
        confidence_per_role=[
            RoleConfidence(role=ThematicRole.OBJECT, confidence=0.2),
            RoleConfidence(role=ThematicRole.GOAL, confidence=0.9),
        ],
    )
    assert instruction.confidence == 0.2


def test_confidence_falls_back_to_the_intent_alone():
    instruction = ParsedInstruction(
        utterance="wake up",
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.MOVE_TO,
        intent_confidence=0.6,
    )
    assert instruction.confidence == 0.6
