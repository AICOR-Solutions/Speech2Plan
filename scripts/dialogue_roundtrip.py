#!/usr/bin/env python3
"""Drive the node through a full dialogue on the local graph, with no robot.

Runs the node, a stand-in for the robot's communication skills, and a probe that
plays the ROS4HRI side: one tracked person, one voice, one utterance. With no
argument the parse is canned, so nothing calls a model; with an utterance the
real parser runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

import rclpy
from communication_skills.action import Ask, Say
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.dialogue import Clarifier
from speech2plan.grounding import Grounder
from speech2plan.ros.intent_node import Speech2PlanNode, Topic
from speech2plan.ros.skill_client import Ros4HriDialogueSkills
from speech2plan.world_source import PrototypeWard

PERSON_ID = "person_demo"
VOICE_ID = "voice_demo"
ANSWERS = {"object": "living room", "goal": "first floor", "confirmed": True}
"""What the stand-in person answers, by the key the question asks for."""

# %% the topics the robot would publish


class HumansTopic:
    """The ROS4HRI topics the robot would publish, which libhri reads."""

    TRACKED_VOICES = "/humans/voices/tracked"
    TRACKED_PERSONS = "/humans/persons/tracked"

    @staticmethod
    def speech(voice_id: str) -> str:
        return f"/humans/voices/{voice_id}/speech"

    @staticmethod
    def voice_of(person_id: str) -> str:
        return f"/humans/persons/{person_id}/voice_id"


# %% the robot's side


class SkillServer(Node):
    """Serves /skill/say and /skill/ask from a script of answers."""

    def __init__(self, answers: Dict[str, str]):
        super().__init__("fake_communication_skills")
        self.answers = answers
        group = ReentrantCallbackGroup()
        ActionServer(self, Say, "/skill/say", self.on_say, callback_group=group)
        ActionServer(self, Ask, "/skill/ask", self.on_ask, callback_group=group)

    def pick(self, offered: list, hint):
        """The offered option the person's words single out, or the answer itself."""
        for option in offered:
            if hint in option:
                return option
        return hint

    def on_say(self, goal):
        print(f"robot : {goal.request.input}")
        goal.succeed()
        return Say.Result()

    def on_ask(self, goal):
        print(f"robot : {goal.request.question}")
        requested = json.loads(goal.request.answers_schema)
        answered = {
            key: self.pick(schema.get("enum", []), self.answers[key])
            for key, schema in requested.items()
            if key in self.answers
        }
        print(f"person: {answered}")
        goal.succeed()
        result = Ask.Result()
        result.answers = json.dumps(answered)
        return result


# %% the ROS4HRI side


class Probe(Node):
    """Announces a tracked person, says one thing, and waits for the intent."""

    def __init__(self):
        super().__init__("speech2plan_probe")
        self.arrived = threading.Event()
        self.intent = None
        self.create_subscription(Intent, Topic.INTENTS, self.on_intent, 10)
        self.voices = self.create_publisher(IdsList, HumansTopic.TRACKED_VOICES, 10)
        self.persons = self.create_publisher(IdsList, HumansTopic.TRACKED_PERSONS, 10)
        self.binding = self.create_publisher(
            String,
            HumansTopic.voice_of(PERSON_ID),
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        """Latched, as `hri_person_manager` publishes it."""

        self.speech = self.create_publisher(
            LiveSpeech, HumansTopic.speech(VOICE_ID), 10
        )

    def on_intent(self, message: Intent) -> None:
        self.intent = message
        self.arrived.set()

    def announce(self, timeout: float = 20.0) -> bool:
        """Put one person and one voice on the graph, and bind them.

        The binding goes out first, as the person manager would publish it long
        before this node starts. The lists then go out until the node has
        subscribed to this voice, because the per-voice topics only exist once
        it has seen the list.
        """
        self.binding.publish(String(data=VOICE_ID))
        deadline = time.time() + timeout
        while time.time() < deadline and not self.subscribed():
            self.voices.publish(IdsList(ids=[VOICE_ID]))
            self.persons.publish(IdsList(ids=[PERSON_ID]))
            time.sleep(0.5)
        time.sleep(1.0)
        return self.subscribed()

    def subscribed(self) -> bool:
        """True when the node listens to this voice."""
        return self.speech.get_subscription_count() > 0

    def utter(self, utterance: str) -> None:
        message = LiveSpeech()
        message.final = utterance
        self.speech.publish(message)
        print(f"person: {utterance}")


# %% the parse, without a model


@dataclass
class CannedParser:
    """Returns the parse of "bring the cup to the kitchen" without calling out."""

    speaker: str = field(kw_only=True, default=PERSON_ID)

    def parse(self, utterance: str, speaker: str = "") -> ParsedInstruction:
        roles = [ThematicRole.OBJECT, ThematicRole.GOAL]
        return ParsedInstruction(
            utterance=utterance,
            speaker=speaker,
            intent=IntentName.BRING_OBJECT,
            roles=[
                RoleBinding(
                    role=ThematicRole.OBJECT,
                    constraint=EntityConstraint(
                        entity_type="Cup", spoken_form="the cup"
                    ),
                ),
                RoleBinding(
                    role=ThematicRole.GOAL,
                    constraint=EntityConstraint(
                        entity_type="Kitchen", spoken_form="the kitchen"
                    ),
                ),
            ],
            intent_confidence=0.9,
            confidence_per_role=[
                RoleConfidence(role=role, confidence=0.9) for role in roles
            ],
        )


# %% entry point


def main() -> None:
    utterance = " ".join(sys.argv[1:])
    world = PrototypeWard().load()
    grounder = Grounder(world=world)

    rclpy.init()
    node = Speech2PlanNode(parser=build_parser(utterance, world), grounder=grounder)
    node.clarifier = Clarifier(
        skills=Ros4HriDialogueSkills(node=node), grounder=grounder
    )
    server = SkillServer(ANSWERS)
    probe = Probe()

    executor = MultiThreadedExecutor()
    for member in (node, server, probe):
        executor.add_node(member)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not probe.announce():
        print("the node never subscribed to the voice")
    probe.utter(utterance or "bring the cup to the kitchen")
    if probe.arrived.wait(timeout=180.0):
        print(f"intent: {probe.intent.intent}")
        print(f"data  : {probe.intent.data}")
        print(f"source: {probe.intent.source}")
    else:
        print("no intent arrived")

    rclpy.shutdown()
    node.listener = None


def build_parser(utterance: str, world):
    """The real parser when an utterance was given, the canned one otherwise."""
    if not utterance:
        return CannedParser()
    from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
    from speech2plan.parsing.instruction_parser import InstructionParser
    from speech2plan.vocabulary import Vocabulary

    return InstructionParser(
        model=ClaudeCommandLineModel(), vocabulary=Vocabulary.from_world(world)
    )


if __name__ == "__main__":
    main()
