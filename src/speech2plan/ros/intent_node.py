from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import rclpy
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from speech2plan.contract import ParsedInstruction, ThematicRole
from speech2plan.parsing.instruction_parser import InstructionParser

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
    """Settings that change how the node behaves, not what it is."""

    minimum_confidence: float = field(kw_only=True, default=0.0)
    queue_depth: int = field(kw_only=True, default=10)


class Speech2PlanNode(Node):
    """Publishes hri_actions_msgs/Intent for what a tracked person says.

    Follows the ROS4HRI tracked lists rather than one fixed voice, so ids that
    appear and disappear are handled. The person id of the speaker travels with
    the intent as its `source`, which is what REP-155 requires.

    Gaze, pointing and engagement are not read yet. An utterance whose object
    is deictic stays unfilled and the intent says so.
    """

    def __init__(
        self,
        parser: InstructionParser,
        configuration: IntentNodeConfiguration = IntentNodeConfiguration(),
    ):
        super().__init__("speech2plan")
        self.parser = parser
        self.configuration = configuration
        self.voice_of_person: Dict[str, str] = {}
        self.speech_subscriptions: Dict[str, object] = {}
        self.person_subscriptions: Dict[str, object] = {}

        reliable = QoSProfile(
            depth=configuration.queue_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.intent_publisher = self.create_publisher(Intent, Topic.INTENTS, reliable)
        self.create_subscription(
            IdsList, Topic.TRACKED_VOICES, self.on_tracked_voices, reliable
        )
        self.create_subscription(
            IdsList, Topic.TRACKED_PERSONS, self.on_tracked_persons, reliable
        )
        self.reliable = reliable
        self.get_logger().info("following ROS4HRI voices, publishing to /intents")

    # %% tracked list bookkeeping

    def on_tracked_voices(self, message: IdsList) -> None:
        live = set(message.ids)
        for voice_id in live - self.speech_subscriptions.keys():
            self.speech_subscriptions[voice_id] = self.create_subscription(
                LiveSpeech,
                Topic.voice_speech(voice_id),
                lambda speech, identifier=voice_id: self.on_speech(identifier, speech),
                self.reliable,
            )
            self.get_logger().info(f"tracking voice {voice_id}")
        for voice_id in self.speech_subscriptions.keys() - live:
            self.destroy_subscription(self.speech_subscriptions.pop(voice_id))

    def on_tracked_persons(self, message: IdsList) -> None:
        live = set(message.ids)
        for person_id in live - self.person_subscriptions.keys():
            self.person_subscriptions[person_id] = self.create_subscription(
                String,
                Topic.person_voice(person_id),
                lambda voice, identifier=person_id: self.on_person_voice(
                    identifier, voice
                ),
                self.reliable,
            )
        for person_id in self.person_subscriptions.keys() - live:
            self.destroy_subscription(self.person_subscriptions.pop(person_id))
            self.voice_of_person.pop(person_id, None)

    def on_person_voice(self, person_id: str, message: String) -> None:
        self.voice_of_person[person_id] = message.data

    def speaker_of(self, voice_id: str) -> str:
        for person_id, bound_voice in self.voice_of_person.items():
            if bound_voice == voice_id:
                return person_id
        return Intent.UNKNOWN_AGENT

    # %% speech to intent

    def on_speech(self, voice_id: str, message: LiveSpeech) -> None:
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
        self.intent_publisher.publish(self.to_message(instruction))
        self.get_logger().info(f"{speaker} said '{message.final}' -> {instruction.intent}")

    def to_message(self, instruction: ParsedInstruction) -> Intent:
        message = Intent()
        message.intent = str(instruction.intent)
        message.data = json.dumps(self.role_data(instruction))
        message.source = instruction.speaker
        message.modality = Intent.MODALITY_SPEECH
        message.confidence = float(instruction.confidence)
        return message

    def role_data(self, instruction: ParsedInstruction) -> dict:
        data = {
            str(binding.role): binding.constraint.entity_type
            for binding in instruction.roles
        }
        data[str(ThematicRole.INPUT)] = instruction.utterance
        if instruction.locale:
            data[str(ThematicRole.LOCALE)] = instruction.locale
        return data


# %% entry point


def main(arguments: Optional[list] = None) -> None:
    from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
    from speech2plan.vocabulary import Vocabulary

    rclpy.init(args=arguments)
    parser = InstructionParser(
        model=ClaudeCommandLineModel(),
        vocabulary=Vocabulary.from_world(load_world()),
    )
    node = Speech2PlanNode(parser)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def load_world():
    raise NotImplementedError(
        "Wire a semantic_digital_twin World here. Until then run the node with "
        "a Vocabulary built by hand, as scripts/parse_once.py does."
    )
