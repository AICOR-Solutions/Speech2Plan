from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import List

import pytest
import rclpy

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.dialogue import Clarifier, DialogueSkills
from speech2plan.exceptions import SkillCallFailed
from speech2plan.grounding import Grounder
from speech2plan.ros.intent_node import Speech2PlanNode
from speech2plan.world_source import PrototypeWard

from .fake_hri import FakeListener, FakePerson, FakeVoice
from .fake_skills import FakeDialogueSkills

# %% stand-ins


@dataclass
class FakeParser:
    """Returns one canned parse, whatever is said."""

    instruction: ParsedInstruction = field(kw_only=True)

    def parse(self, utterance: str, speaker: str = "") -> ParsedInstruction:
        return replace(self.instruction, utterance=utterance, speaker=speaker)


@dataclass
class Recorder:
    """Keeps the intents the node would have published."""

    published: List[object] = field(kw_only=True, default_factory=list)

    def publish(self, message) -> None:
        self.published.append(message)


class BrokenDialogueSkills(DialogueSkills):
    """A robot with no communication skills running."""

    def say(self, text: str, person_id: str) -> None:
        raise SkillCallFailed(skill="/skill/say", reason="no action server")

    def ask(self, question):
        raise SkillCallFailed(skill="/skill/ask", reason="no action server")


# %% fixtures


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


def ambiguous_object() -> ParsedInstruction:
    """Bring the cup, of which two stand on the speaker's floor."""
    return ParsedInstruction(
        utterance="",
        speaker="",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT, constraint=EntityConstraint(entity_type="Cup")
            )
        ],
        intent_confidence=0.9,
        confidence_per_role=[
            RoleConfidence(role=ThematicRole.OBJECT, confidence=0.9)
        ],
    )


def doubted_object() -> ParsedInstruction:
    """Take the table, which grounds, scored too low to act on."""
    return ParsedInstruction(
        utterance="",
        speaker="",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT, constraint=EntityConstraint(entity_type="Table")
            )
        ],
        intent_confidence=0.2,
    )


def node_with(ward, skills, instruction=None, listener=None) -> Speech2PlanNode:
    """A node with a stand-in listener, recording instead of publishing."""
    grounder = Grounder(world=ward)
    node = Speech2PlanNode(
        parser=FakeParser(instruction=instruction or ambiguous_object()),
        grounder=grounder,
        clarifier=(
            None
            if skills is None
            else Clarifier(skills=skills, grounder=grounder)
        ),
        listener=listener or FakeListener(),
    )
    node.intent_publisher = Recorder()
    return node


def utter(node, utterance: str, voice_id: str = "voice_1") -> None:
    """Track a voice and let it speak, the way libhri would."""
    node.listener.track(FakeVoice(id=voice_id)).say(utterance)


# %% who spoke


def test_a_voice_bound_to_a_person_is_attributed(ros, ward):
    voice = FakeVoice(id="voice_1")
    listener = FakeListener(
        tracked_persons={"person_abc": FakePerson(id="person_abc", voice=voice)}
    )
    node = node_with(ward, None, listener=listener)

    listener.track(voice).say("bring this to the kitchen")

    published = node.intent_publisher.published
    assert len(published) == 1
    assert published[0].source == "person_abc"
    assert json.loads(published[0].data)["locale"] == "en_US"
    node.destroy_node()


def test_a_voice_nobody_claims_is_unknown(ros, ward):
    listener = FakeListener(tracked_persons={"person_abc": FakePerson(id="person_abc")})
    node = node_with(ward, None, listener=listener)

    utter(node, "bring this to the kitchen")

    assert node.intent_publisher.published[0].source == "__unknown_agent__"
    node.destroy_node()


# %% the dialogue in the node's path


def test_the_answer_reaches_the_published_intent(ros, ward):
    voice = FakeVoice(id="voice_1")
    listener = FakeListener(
        tracked_persons={"person_abc": FakePerson(id="person_abc", voice=voice)}
    )
    skills = FakeDialogueSkills(
        scripted_answers={
            "object": "the one in the living room on the ground floor",
            "confirmed": True,
        }
    )
    node = node_with(ward, skills, listener=listener)

    listener.track(voice).say("bring the cup here")

    assert len(skills.said) == 1
    assert [question.role for question in skills.asked] == [ThematicRole.OBJECT, None]
    assert skills.asked[0].person_id == "person_abc"
    published = node.intent_publisher.published
    assert len(published) == 1
    assert json.loads(published[0].data)["object"] == "living_room_cup"
    node.destroy_node()


def test_without_a_clarifier_an_ambiguous_role_stays_a_constraint(ros, ward):
    node = node_with(ward, None)

    utter(node, "bring the cup here")

    data = json.loads(node.intent_publisher.published[0].data)
    assert "object" not in data
    constraint = data["other_data"]["constraints"][0]
    assert constraint["status"] == "ambiguous"
    assert constraint["candidates"] == ["cup", "living_room_cup"]
    node.destroy_node()


def test_a_refusal_publishes_nothing(ros, ward):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": False})
    node = node_with(ward, skills, doubted_object())

    utter(node, "do the thing")

    assert node.intent_publisher.published == []
    assert skills.asked[-1].text == (
        "I will bring the table in the kitchen on the ground floor to somewhere. "
        "Is that correct?"
    )
    node.destroy_node()


def test_a_missing_skill_server_does_not_stop_the_intent(ros, ward):
    node = node_with(ward, BrokenDialogueSkills())

    utter(node, "bring this to the kitchen")

    assert len(node.intent_publisher.published) == 1
    node.destroy_node()
