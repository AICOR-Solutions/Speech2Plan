from __future__ import annotations

from hri_actions_msgs.msg import Intent

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    RoleBinding,
    ThematicRole,
)
from speech2plan.grounding import Grounder
from speech2plan.ros.intent_node import Speech2PlanNode
from speech2plan.world_source import PrototypeWard

# %% message construction without a ROS runtime


def build_message(utterance: str, entity_type: str, role: ThematicRole) -> Intent:
    """The Intent the node would publish for one parsed role."""
    world = PrototypeWard().load()
    instruction = ParsedInstruction(
        utterance=utterance,
        speaker=Intent.UNKNOWN_AGENT,
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=role, constraint=EntityConstraint(entity_type=entity_type)
            )
        ],
    )
    grounder = Grounder(world=world)
    return Speech2PlanNode.to_message(
        None, instruction, grounder.ground(instruction)
    )


def test_a_resolved_role_carries_the_entity_name():
    message = build_message("take the plate", "Plate", ThematicRole.OBJECT)
    assert '"object": "plate"' in message.data


def test_an_ambiguous_role_falls_back_to_the_entity_type():
    message = build_message("go to the kitchen", "Kitchen", ThematicRole.GOAL)
    assert '"goal": "Kitchen"' in message.data


def test_the_utterance_travels_as_input():
    message = build_message("take the plate", "Plate", ThematicRole.OBJECT)
    assert '"input": "take the plate"' in message.data
    assert message.modality == Intent.MODALITY_SPEECH
