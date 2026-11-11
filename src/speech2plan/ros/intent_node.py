from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Optional

import rclpy
from hri import HRIListener, Voice
from hri_actions_msgs.msg import Intent
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

from speech2plan.contract import IntentName, ParsedInstruction
from speech2plan.dialogue import Clarification, Clarifier
from speech2plan.exceptions import (
    LanguageModelCallFailed,
    SkillCallFailed,
    UnusableReply,
)
from speech2plan.grounding import Grounder
from speech2plan.intent_message import PublishedIntent
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.ros.task_submission import TaskSubmission
from speech2plan.settings import Words, default_settings

# %% what the node says about handing a task on


class SubmissionNotice(StrEnum):
    """What the robot says when the task the speaker confirmed is not taken on."""

    NOT_TAKEN = "I cannot take on that task right now."


# %% node


@dataclass
class IntentNodeConfiguration:
    """Settings that change how the node behaves."""

    minimum_confidence: float = field(kw_only=True, default=0.0)
    priority: int = field(kw_only=True, default=128)
    """The priority every submitted intent carries."""

    humans_period_seconds: float = field(kw_only=True, default=0.05)
    """How often the ROS4HRI graph is read."""

    words: Words = field(kw_only=True, default_factory=lambda: default_settings().words)
    """Which utterances are only filler."""


class Speech2PlanNode(Node):
    """Hands the plan executive an hri_actions_msgs/Intent for what a tracked person
    says, as a `CarryOutIntent` goal."""

    def __init__(
        self,
        parser: InstructionParser,
        grounder: Grounder,
        clarifier: Optional[Clarifier] = None,
        configuration: Optional[IntentNodeConfiguration] = None,
        listener: Optional[HRIListener] = None,
    ):
        super().__init__("speech2plan")
        self.parser = parser
        self.grounder = grounder
        self.clarifier = clarifier
        self.configuration = configuration or IntentNodeConfiguration()
        self.world_synchronizer: Optional[WorldSynchronizer] = None
        """Buffers the changes other processes make to the world, if it is shared."""
        self.handling_instruction = threading.Lock()
        """Held from parsing an instruction until its dialogue ends."""

        self.task_submission = TaskSubmission(
            node=self, priority=self.configuration.priority
        )
        self.listener = listener or HRIListener(
            f"{self.get_name()}_hri_listener", False
        )
        """Follows the ROS4HRI graph: who is tracked, and whose voice is whose."""

        self.listener.on_voice(self.on_voice)
        self.create_timer(
            self.configuration.humans_period_seconds,
            self.follow_humans,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def follow_humans(self) -> None:
        """Read what the ROS4HRI graph published, in this node's own thread.

        A dialogue blocks here while it runs, which is why the listener is not
        left to spin itself: every callback then belongs to one executor.
        """
        self.listener.spin_all(
            timedelta(seconds=self.configuration.humans_period_seconds)
        )

    def on_voice(self, voice: Voice) -> None:
        """Listen to a voice that has just been tracked."""
        voice.on_speech(lambda utterance, heard=voice: self.on_speech(heard, utterance))

    def speaker_of(self, voice: Voice) -> str:
        """The person this voice belongs to, or Intent.UNKNOWN_AGENT."""
        for person_id, person in self.listener.tracked_persons.items():
            if person.voice is not None and person.voice.id == voice.id:
                return person_id
        return Intent.UNKNOWN_AGENT

    def follow_world(self) -> None:
        """Apply the world changes received since the last utterance.

        The parser reads its vocabulary afresh only when something changed.
        """
        if (
            self.world_synchronizer is None
            or not self.world_synchronizer.missed_messages
        ):
            return
        self.world_synchronizer.apply_missed_messages()
        self.parser.follow(self.grounder.world)

    # %% speech to intent

    def on_speech(self, voice: Voice, utterance: str) -> None:
        """Handle a final utterance, unless it is a filler, an answer, or said while
        another instruction is handled."""
        if self.configuration.words.is_filler(utterance):
            return
        if self.clarifier is not None and self.clarifier.skills.took_as_answer(
            utterance
        ):
            return
        if not self.handling_instruction.acquire(blocking=False):
            self.get_logger().info(f"ignoring '{utterance}' during a dialogue")
            return
        try:
            self.handle_instruction(voice, utterance)
        finally:
            self.handling_instruction.release()

    def handle_instruction(self, voice: Voice, utterance: str) -> None:
        """Parse an utterance, clarify it, and hand it to the plan executive."""
        self.follow_world()
        speaker = self.speaker_of(voice)
        try:
            instruction = self.parser.parse(utterance, speaker=speaker)
        except (LanguageModelCallFailed, UnusableReply) as failure:
            self.get_logger().error(f"dropping '{utterance}': {failure}")
            return
        if instruction.intent == IntentName.RAW_USER_INPUT:
            self.get_logger().info(f"no task in '{utterance}'")
            return
        instruction.locale = voice.locale or None
        if instruction.confidence < self.configuration.minimum_confidence:
            self.get_logger().info(
                f"dropping '{utterance}' at confidence {instruction.confidence}"
            )
            return
        clarification = self.clarify(instruction)
        if clarification is None:
            self.get_logger().info(f"'{utterance}' was withdrawn")
            return
        intent = PublishedIntent(
            instruction=clarification.instruction,
            outcomes=clarification.outcomes,
            priority=self.configuration.priority,
        ).as_message()
        try:
            self.task_submission.submit(intent)
        except SkillCallFailed as failure:
            self.get_logger().error(f"nobody carries out '{utterance}': {failure}")
            self.tell(clarification.instruction.speaker, SubmissionNotice.NOT_TAKEN)
            return
        self.get_logger().info(f"{speaker} said '{utterance}' -> {instruction.intent}")

    def tell(self, person_id: str, text: str) -> None:
        """Say `text` to `person_id`, if the node can speak at all."""
        if self.clarifier is None:
            return
        try:
            self.clarifier.skills.say(str(text), person_id)
        except SkillCallFailed as failure:
            self.get_logger().error(f"could not say '{text}': {failure}")

    def clarify(self, instruction: ParsedInstruction) -> Optional[Clarification]:
        """Run the dialogue, or ground in silence when there is no clarifier.

        Returns None when the instruction is withdrawn or cannot be confirmed.
        """
        if self.clarifier is None:
            return Clarification(
                instruction=instruction, outcomes=self.grounder.ground(instruction)
            )
        try:
            clarification = self.clarifier.clarify(instruction)
        except SkillCallFailed as failure:
            self.get_logger().error(f"nobody can confirm, withdrawing: {failure}")
            return None
        return clarification if clarification.confirmed else None


# %% entry point


def main(arguments: Optional[list] = None) -> None:
    """Run the node against a fetched world, falling back to the care ward."""
    from rclpy.executors import MultiThreadedExecutor

    from speech2plan.parsing.answer_interpreter import AnswerInterpreter
    from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
    from speech2plan.ros.fetched_world import FetchedWorld
    from speech2plan.ros.help_requests import HelpRequests
    from speech2plan.ros.skill_client import SpokenAnswerSkills
    from speech2plan.world_source import CareWard

    rclpy.init(args=arguments)
    bootstrap = rclpy.create_node("speech2plan_world_fetch")
    fetched = FetchedWorld(node=bootstrap)
    is_shared = fetched.is_available()
    world = fetched.load() if is_shared else CareWard().load()
    bootstrap.destroy_node()

    model = ClaudeCommandLineModel()
    grounder = Grounder(world=world)
    parser = InstructionParser.for_world(world, model)
    node = Speech2PlanNode(parser=parser, grounder=grounder)
    if is_shared:
        node.world_synchronizer = WorldSynchronizer(
            _world=world, node=node, defer_incoming_updates=True
        )
    node.clarifier = Clarifier(
        skills=SpokenAnswerSkills(node=node),
        grounder=grounder,
        interpreter=AnswerInterpreter(model=model),
    )
    node.help_requests = HelpRequests(
        node=node, skills=node.clarifier.skills, dialogue=node.handling_instruction
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    if rclpy.ok():
        rclpy.shutdown()
    node.listener = None


if __name__ == "__main__":
    main()
