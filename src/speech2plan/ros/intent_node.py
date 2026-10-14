from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import rclpy
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from semantic_digital_twin.world import World
from std_msgs.msg import String

from speech2plan.contract import ParsedInstruction, ThematicRole
from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.world_source import WorldSource

# %% topic names


class Topic:
    """ROS4HRI topics this node uses."""

    TRACKED_VOICES = "/humans/voices/tracked"
    TRACKED_PERSONS = "/humans/persons/tracked"
    INTENTS = "/intents"

    @staticmethod
    def voice_speech(voice_id: str) -> str:
        return f"/humans/voices/{voice_id}/speech"

    @staticmethod
    def person_voice(person_id: str) -> str:
        return f"/humans/persons/{person_id}/voice_id"


# %% node


@dataclass
class IntentNodeConfiguration:
    """Settings that change how the node behaves."""

    minimum_confidence: float = field(kw_only=True, default=0.0)
    queue_depth: int = field(kw_only=True, default=10)


class Speech2PlanNode(Node):
    """Publishes hri_actions_msgs/Intent for what a tracked person says."""

    def __init__(
        self,
        parser: InstructionParser,
        grounder: Grounder,
        configuration: IntentNodeConfiguration = IntentNodeConfiguration(),
    ):
        super().__init__("speech2plan")
        self.parser = parser
        self.grounder = grounder
        self.configuration = configuration
        self.voice_of_person: Dict[str, str] = {}
        self.speech_subscriptions: Dict[str, object] = {}
        self.person_subscriptions: Dict[str, object] = {}
        self.reliable = QoSProfile(
            depth=configuration.queue_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.intent_publisher = self.create_publisher(
            Intent, Topic.INTENTS, self.reliable
        )
        self.create_subscription(
            IdsList, Topic.TRACKED_VOICES, self.on_tracked_voices, self.reliable
        )
        self.create_subscription(
            IdsList, Topic.TRACKED_PERSONS, self.on_tracked_persons, self.reliable
        )

    # %% tracked list bookkeeping

    def on_tracked_voices(self, message: IdsList) -> None:
        """Subscribe to new voices and drop the ones that went away."""
        live = set(message.ids)
        for voice_id in live - self.speech_subscriptions.keys():
            self.speech_subscriptions[voice_id] = self.create_subscription(
                LiveSpeech,
                Topic.voice_speech(voice_id),
                lambda speech, identifier=voice_id: self.on_speech(identifier, speech),
                self.reliable,
            )
        for voice_id in self.speech_subscriptions.keys() - live:
            self.destroy_subscription(self.speech_subscriptions.pop(voice_id))

    def on_tracked_persons(self, message: IdsList) -> None:
        """Follow the voice binding of every tracked person."""
        live = set(message.ids)
        for person_id in live - self.person_subscriptions.keys():
            self.person_subscriptions[person_id] = self.create_subscription(
                String,
                Topic.person_voice(person_id),
                lambda voice, identifier=person_id: self.voice_of_person.__setitem__(
                    identifier, voice.data
                ),
                self.reliable,
            )
        for person_id in self.person_subscriptions.keys() - live:
            self.destroy_subscription(self.person_subscriptions.pop(person_id))
            self.voice_of_person.pop(person_id, None)

    def speaker_of(self, voice_id: str) -> str:
        """The person bound to `voice_id`, or Intent.UNKNOWN_AGENT."""
        for person_id, bound_voice in self.voice_of_person.items():
            if bound_voice == voice_id:
                return person_id
        return Intent.UNKNOWN_AGENT

    # %% speech to intent

    def on_speech(self, voice_id: str, message: LiveSpeech) -> None:
        """Parse a final utterance and publish it as an intent."""
        if not message.final:
            return
        speaker = self.speaker_of(voice_id)
        instruction = self.parser.parse(message.final, speaker=speaker)
        instruction.locale = message.locale or None
        if instruction.confidence < self.configuration.minimum_confidence:
            self.get_logger().info(
                f"dropping '{message.final}' at confidence {instruction.confidence}"
            )
            return
        outcomes = self.grounder.ground(instruction)
        for outcome in outcomes:
            if isinstance(outcome, Ambiguity):
                names = ", ".join(str(entity.name) for entity in outcome.candidates)
                self.get_logger().info(f"{outcome.role} is ambiguous between {names}")
        self.intent_publisher.publish(self.to_message(instruction, outcomes))
        self.get_logger().info(
            f"{speaker} said '{message.final}' -> {instruction.intent}"
        )

    def to_message(self, instruction: ParsedInstruction, outcomes) -> Intent:
        """The instruction as an hri_actions_msgs/Intent, grounded where possible."""
        resolved = {
            str(outcome.role): str(outcome.entity.name)
            for outcome in outcomes
            if isinstance(outcome, ResolvedRole)
        }
        data = {
            str(binding.role): resolved.get(
                str(binding.role), binding.constraint.entity_type
            )
            for binding in instruction.roles
        }
        data[str(ThematicRole.INPUT)] = instruction.utterance
        if instruction.locale:
            data[str(ThematicRole.LOCALE)] = instruction.locale

        message = Intent()
        message.intent = str(instruction.intent)
        message.data = json.dumps(data)
        message.source = instruction.speaker
        message.modality = Intent.MODALITY_SPEECH
        message.confidence = float(instruction.confidence)
        return message


# %% entry point


def load_world(preferred: WorldSource, fallback: WorldSource) -> World:
    """Load from `preferred`, or from `fallback` when nothing serves a world."""
    if preferred.is_available():
        return preferred.load()
    return fallback.load()


def main(arguments: Optional[list] = None) -> None:
    """Run the node against a fetched world, falling back to the prototype."""
    from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
    from speech2plan.ros.fetched_world import FetchedWorld
    from speech2plan.vocabulary import Vocabulary
    from speech2plan.world_source import PrototypeWard

    rclpy.init(args=arguments)
    bootstrap = rclpy.create_node("speech2plan_world_fetch")
    world = load_world(FetchedWorld(node=bootstrap), PrototypeWard())
    bootstrap.destroy_node()

    node = Speech2PlanNode(
        parser=InstructionParser(
            model=ClaudeCommandLineModel(), vocabulary=Vocabulary.from_world(world)
        ),
        grounder=Grounder(world=world),
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
