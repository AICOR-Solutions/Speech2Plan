from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List

from communication_skills.action import Say
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from std_skills.msg import Meta

from speech2plan.dialogue import DialogueSkills, Question
from speech2plan.exceptions import SkillCallFailed
from speech2plan.settings import Words, default_settings

# %% skill paths


class SkillPath(StrEnum):
    """Where the ROS4HRI communication skills live."""

    SAY = "/skill/say"


class HumansTopic(StrEnum):
    """Where ROS4HRI publishes who is tracked, whose voice is whose, and what each
    voice says."""

    TRACKED_VOICES = "/humans/voices/tracked"
    TRACKED_PERSONS = "/humans/persons/tracked"
    SPEECH = "/humans/voices/{voice_id}/speech"
    VOICE_OF_PERSON = "/humans/persons/{person_id}/voice_id"


# %% waiting on ROS


def completes(future: Future, timeout_seconds: float) -> bool:
    """Whether `future` completes within `timeout_seconds`, completed by an executor
    on another thread."""
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout=timeout_seconds)


# %% speaking and listening


@dataclass
class SpokenAnswerSkills(DialogueSkills):
    """Says through the ROS4HRI `say` skill, and takes the next sentence a tracked
    voice speaks as the answer to a question.

    TIAGo's `ask` skill is not used: its only chatbot, Rasa, cannot fill an answer
    schema. A yes or no question is answered by the words of that sentence, any
    other question by the sentence itself.
    """

    node: Node = field(kw_only=True)
    priority: int = field(kw_only=True, default=Meta.NORMAL_PRIORITY)
    timeout_seconds: float = field(kw_only=True, default=10.0)
    """How long to wait for the say skill to be there."""

    answer_timeout_seconds: float = field(kw_only=True, default=120.0)
    """How long to wait for a sentence to be said, or a person to answer."""

    words: Words = field(kw_only=True, default_factory=lambda: default_settings().words)
    """Which sentences are filler, and which words say yes or no."""

    def __post_init__(self):
        self.say_client = ActionClient(
            self.node, Say, SkillPath.SAY, callback_group=ReentrantCallbackGroup()
        )
        self.speech_group = ReentrantCallbackGroup()
        self.speech_subscriptions: Dict[str, object] = {}
        self.listening = False
        self.answer = ""
        self.answer_heard = threading.Event()
        self.answers_taken: List[str] = []
        """Sentences taken as answers, which the node must not parse again."""
        self.node.create_subscription(
            IdsList,
            HumansTopic.TRACKED_VOICES,
            self.follow_voices,
            10,
            callback_group=self.speech_group,
        )

    # %% saying

    def say(self, text: str, person_id: str) -> None:
        """Say `text` to `person_id`, and wait until it is said.

        Waiting is explicit rather than `send_goal`, because this runs in a
        subscription callback and must not spin the node it is called from.
        """
        if not self.say_client.wait_for_server(timeout_sec=self.timeout_seconds):
            raise SkillCallFailed(skill=SkillPath.SAY, reason="no action server")
        goal = Say.Goal()
        goal.meta.caller = self.node.get_name()
        goal.meta.priority = self.priority
        goal.person_id = "" if person_id == Intent.UNKNOWN_AGENT else person_id
        goal.input = text
        sent = self.say_client.send_goal_async(goal)
        if not completes(sent, self.answer_timeout_seconds):
            sent.cancel()
            raise SkillCallFailed(skill=SkillPath.SAY, reason="no answer in time")
        handle = sent.result()
        if not handle.accepted:
            raise SkillCallFailed(skill=SkillPath.SAY, reason="goal rejected")
        if not completes(handle.get_result_async(), self.answer_timeout_seconds):
            handle.cancel_goal_async()
            raise SkillCallFailed(skill=SkillPath.SAY, reason="no result in time")

    # %% listening

    def follow_voices(self, tracked: IdsList) -> None:
        """Listen to every voice ROS4HRI tracks."""
        for voice_id in tracked.ids:
            if voice_id in self.speech_subscriptions:
                continue
            self.speech_subscriptions[voice_id] = self.node.create_subscription(
                LiveSpeech,
                HumansTopic.SPEECH.format(voice_id=voice_id),
                self.on_speech,
                10,
                callback_group=self.speech_group,
            )

    def on_speech(self, speech: LiveSpeech) -> None:
        if not self.listening or self.words.is_filler(speech.final):
            return
        self.listening = False
        self.answer = speech.final
        self.answers_taken.append(speech.final)
        self.answer_heard.set()

    def ask(self, question: Question) -> Dict[str, object]:
        self.say(question.text, question.person_id)
        self.answer_heard.clear()
        self.listening = True
        if not self.answer_heard.wait(timeout=self.answer_timeout_seconds):
            self.listening = False
            raise SkillCallFailed(skill="spoken answer", reason="no answer in time")
        return self.answers_from(question.answers_schema, self.answer)

    def answers_from(self, answers_schema: Dict[str, dict], spoken: str) -> Dict:
        """`spoken` under the key the question asked for, a yes or no as a boolean."""
        key = next(iter(answers_schema))
        if answers_schema[key].get("type") != "boolean":
            return {key: spoken}
        words = set(self.words.of(spoken))
        if words & self.words.negative:
            return {key: False}
        if words & self.words.affirmative:
            return {key: True}
        return {}

    def took_as_answer(self, utterance: str) -> bool:
        if utterance not in self.answers_taken:
            return False
        self.answers_taken.remove(utterance)
        return True
