#!/usr/bin/env python3
"""Drive the node through a full dialogue on the local graph, with no robot.

Runs the node, a stand-in for the robot's say skill, and a probe that plays the
ROS4HRI side, one tracked person, one voice, one utterance and the answers to the
robot's questions, and the plan executive, taking the task the node submits. With no
argument the parse is canned, so nothing calls a model; with an utterance the
real parser runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import rclpy
from communication_skills.action import Say
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action.server import ServerGoalHandle
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from speech2plan_msgs.action import CarryOutIntent
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
from speech2plan.ros.intent_node import Speech2PlanNode
from speech2plan.ros.skill_client import (
    HumansTopic,
    SkillPath,
    SpokenAnswerSkills,
)
from speech2plan.ros.task_submission import TaskSkillPath
from speech2plan.world_source import PrototypeWard

# %% the robot's side


class SkillServer(Node):
    """Serves /skill/say, and answers each question the robot says, in turn, by
    speaking as the person would once the robot listens."""

    def __init__(self, speak: Callable[[str], None], answers: Optional[List[str]] = None):
        super().__init__("fake_say_skill")
        self.speak = speak
        self.answers = answers or ["yes"]
        """What the person answers, in turn; the last one again once the rest are said."""

        ActionServer(
            self, Say, SkillPath.SAY, self.on_say, callback_group=ReentrantCallbackGroup()
        )

    def on_say(self, goal):
        text = goal.request.input
        print(f"robot : {text}")
        if text.endswith("?"):
            answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            threading.Timer(0.5, self.speak, args=[answer]).start()
        goal.succeed()
        return Say.Result()


# %% the ROS4HRI side


class Probe(Node):
    """Announces a tracked person, says one thing, and takes the task submitted for
    it, as the plan executive would."""

    def __init__(self, person_id: str = "person_demo", voice_id: str = "voice_demo"):
        super().__init__("speech2plan_probe")
        self.person_id = person_id
        self.voice_id = voice_id
        self.arrived = threading.Event()
        self.intent = None
        ActionServer(
            self,
            CarryOutIntent,
            str(TaskSkillPath.CARRY_OUT_INTENT),
            self.on_task,
            callback_group=ReentrantCallbackGroup(),
        )
        self.voices = self.create_publisher(IdsList, HumansTopic.TRACKED_VOICES, 10)
        self.persons = self.create_publisher(IdsList, HumansTopic.TRACKED_PERSONS, 10)
        self.binding = self.create_publisher(
            String,
            HumansTopic.VOICE_OF_PERSON.format(person_id=person_id),
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        """Latched, as `hri_person_manager` publishes it."""

        self.speech = self.create_publisher(
            LiveSpeech, HumansTopic.SPEECH.format(voice_id=voice_id), 10
        )

    def on_task(self, goal: ServerGoalHandle) -> CarryOutIntent.Result:
        self.intent = goal.request.intent
        self.arrived.set()
        goal.succeed()
        return CarryOutIntent.Result()

    def announce(self, timeout: float = 20.0) -> bool:
        """Put one person and one voice on the graph, and bind them.

        The binding goes out first, as the person manager would publish it long
        before this node starts. The lists then go out until the node has
        subscribed to this voice, because the per-voice topics only exist once
        it has seen the list.
        """
        self.binding.publish(String(data=self.voice_id))
        deadline = time.time() + timeout
        while time.time() < deadline and not self.subscribed():
            self.voices.publish(IdsList(ids=[self.voice_id]))
            self.persons.publish(IdsList(ids=[self.person_id]))
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
                        entity_type=None, spoken_form="the cup"
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
    node.clarifier = Clarifier(skills=SpokenAnswerSkills(node=node), grounder=grounder)
    probe = Probe()
    server = SkillServer(speak=probe.utter)

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

    return InstructionParser.for_world(world, ClaudeCommandLineModel())


if __name__ == "__main__":
    main()
