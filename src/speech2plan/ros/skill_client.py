from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from communication_skills.action import Ask, Say
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_skills.msg import Meta

from speech2plan.dialogue import DialogueSkills, Question, is_filler
from speech2plan.exceptions import SkillCallFailed

# %% skill paths


class SkillPath:
    """Where the ROS4HRI communication skills live."""

    SAY = "/skill/say"
    ASK = "/skill/ask"


class VoiceTopic:
    """Where ROS4HRI publishes the tracked voices and what they say."""

    TRACKED = "/humans/voices/tracked"
    SPEECH = "/humans/voices/{voice_id}/speech"


# %% client


@dataclass
class Ros4HriDialogueSkills(DialogueSkills):
    """Calls the ROS4HRI say and ask skills."""

    node: Node = field(kw_only=True)
    priority: int = field(kw_only=True, default=Meta.NORMAL_PRIORITY)
    timeout_seconds: float = field(kw_only=True, default=10.0)
    """How long to wait for a skill server before giving up on it."""

    answer_timeout_seconds: float = field(kw_only=True, default=120.0)
    """How long to wait for a person to answer."""

    def __post_init__(self):
        group = ReentrantCallbackGroup()
        self.say_client = ActionClient(
            self.node, Say, SkillPath.SAY, callback_group=group
        )
        self.ask_client = ActionClient(
            self.node, Ask, SkillPath.ASK, callback_group=group
        )

    def meta(self) -> Meta:
        """The call metadata every skill goal carries."""
        meta = Meta()
        meta.caller = self.node.get_name()
        meta.priority = self.priority
        return meta

    def addressee(self, person_id: str) -> str:
        """The skill's person id, empty for whoever is there."""
        return "" if person_id == Intent.UNKNOWN_AGENT else person_id

    def say(self, text: str, person_id: str) -> None:
        goal = Say.Goal()
        goal.meta = self.meta()
        goal.person_id = self.addressee(person_id)
        goal.input = text
        self.send(self.say_client, goal, SkillPath.SAY)

    def ask(self, question: Question) -> Dict[str, object]:
        goal = Ask.Goal()
        goal.meta = self.meta()
        goal.person_id = self.addressee(question.person_id)
        goal.question = question.text
        goal.answers_schema = json.dumps(question.answers_schema)
        result = self.send(self.ask_client, goal, SkillPath.ASK)
        return json.loads(result.answers)

    def send(self, client: ActionClient, goal, path: str):
        """Send a goal and wait for its result.

        The waiting is explicit rather than `send_goal`, because this runs in a
        subscription callback and must not spin the node it is called from.
        """
        if not client.wait_for_server(timeout_sec=self.timeout_seconds):
            raise SkillCallFailed(skill=path, reason="no action server")
        handle = self.wait_for(client.send_goal_async(goal), path)
        if not handle.accepted:
            raise SkillCallFailed(skill=path, reason="goal rejected")
        result = handle.get_result_async()
        if not self.completes(result):
            handle.cancel_goal_async()
            raise SkillCallFailed(skill=path, reason="no result in time")
        return result.result().result

    def wait_for(self, future, path: str):
        """The result of a future that the executor completes on another thread."""
        if not self.completes(future):
            future.cancel()
            raise SkillCallFailed(skill=path, reason="no result in time")
        return future.result()

    def completes(self, future) -> bool:
        """Whether `future` completes within the answer timeout."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return done.wait(timeout=self.answer_timeout_seconds)


# %% answers read from speech


@dataclass
class SpokenAnswerSkills(Ros4HriDialogueSkills):
    """Asks through the say skill and takes the next sentence a tracked voice speaks
    as the answer.

    Stands in for an ask skill whose chatbot cannot fill an answer schema, as on
    TIAGo with Rasa. A yes or no question is answered by the words of that sentence,
    any other question by the sentence itself.
    """

    affirmative_words: Tuple[str, ...] = field(
        kw_only=True,
        default=("yes", "yeah", "yep", "sure", "correct", "right", "ok", "okay"),
    )
    negative_words: Tuple[str, ...] = field(
        kw_only=True, default=("no", "nope", "not", "wrong")
    )

    def __post_init__(self):
        super().__post_init__()
        self.speech_group = ReentrantCallbackGroup()
        self.speech_subscriptions: Dict[str, object] = {}
        self.listening = False
        self.answer = ""
        self.answer_heard = threading.Event()
        self.answers_taken: List[str] = []
        """Sentences taken as answers, which the node must not parse again."""
        self.node.create_subscription(
            IdsList,
            VoiceTopic.TRACKED,
            self.follow_voices,
            10,
            callback_group=self.speech_group,
        )

    def follow_voices(self, tracked: IdsList) -> None:
        """Listen to every voice ROS4HRI tracks."""
        for voice_id in tracked.ids:
            if voice_id in self.speech_subscriptions:
                continue
            self.speech_subscriptions[voice_id] = self.node.create_subscription(
                LiveSpeech,
                VoiceTopic.SPEECH.format(voice_id=voice_id),
                self.on_speech,
                10,
                callback_group=self.speech_group,
            )

    def on_speech(self, speech: LiveSpeech) -> None:
        if not self.listening or is_filler(speech.final):
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
        words = set(re.findall(r"[a-z']+", spoken.lower()))
        if words & set(self.negative_words):
            return {key: False}
        if words & set(self.affirmative_words):
            return {key: True}
        return {}

    def took_as_answer(self, utterance: str) -> bool:
        if utterance not in self.answers_taken:
            return False
        self.answers_taken.remove(utterance)
        return True
