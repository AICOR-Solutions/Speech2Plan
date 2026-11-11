from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
import rclpy
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.task import Future

from speech2plan.dialogue import AnswerKey, Question
from speech2plan.exceptions import SkillCallFailed
from speech2plan.ros.skill_client import Ros4HriDialogueSkills, SpokenAnswerSkills

# %% stand-ins


@dataclass
class UnansweredGoal:
    """An accepted goal whose result never arrives."""

    accepted: bool = field(kw_only=True, default=True)
    cancelled: bool = field(kw_only=True, default=False)

    def get_result_async(self) -> Future:
        return Future()

    def cancel_goal_async(self) -> Future:
        self.cancelled = True
        return Future()


@dataclass
class ServerWithUnansweredGoals:
    """An action server that accepts every goal and never finishes one."""

    goal: UnansweredGoal = field(kw_only=True, default_factory=UnansweredGoal)

    def wait_for_server(self, timeout_sec: float) -> bool:
        return True

    def send_goal_async(self, goal) -> Future:
        accepted = Future()
        accepted.set_result(self.goal)
        return accepted


@dataclass
class QuietSpokenAnswerSkills(SpokenAnswerSkills):
    """Spoken answers, with what the robot says recorded instead of spoken."""

    said: list = field(kw_only=True, default_factory=list)

    def say(self, text: str, person_id: str) -> None:
        self.said.append(text)


def answer_later(skills: SpokenAnswerSkills, *sentences: str) -> threading.Thread:
    """Speak `sentences` in turn once the skills are listening, as a person would."""

    def speak():
        while not skills.listening:
            threading.Event().wait(0.01)
        for sentence in sentences:
            skills.on_speech(LiveSpeech(final=sentence))

    speaker = threading.Thread(target=speak)
    speaker.start()
    return speaker


def confirmation() -> Question:
    return Question(
        text="Is that correct?",
        person_id="person_abc",
        answers_schema={AnswerKey.CONFIRMED: {"type": "boolean"}},
    )


# %% fixtures


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def spoken(ros):
    node = rclpy.create_node("spoken_answers_under_test")
    yield QuietSpokenAnswerSkills(node=node, answer_timeout_seconds=2.0)
    node.destroy_node()


# %% a question nobody answers


def test_an_unanswered_question_is_cancelled_on_the_server(ros):
    node = rclpy.create_node("skill_client_under_test")
    skills = Ros4HriDialogueSkills(node=node, answer_timeout_seconds=0.1)
    skills.ask_client = ServerWithUnansweredGoals()

    with pytest.raises(SkillCallFailed):
        skills.ask(
            Question(
                text="Is that correct?",
                person_id="person_abc",
                answers_schema={"confirmed": {"type": "boolean"}},
            )
        )

    assert skills.ask_client.goal.cancelled
    node.destroy_node()


# %% answers read from speech


@pytest.mark.parametrize(
    "sentence, confirmed",
    [("he yes", True), ("yeah that's right", True), ("no that is wrong", False)],
)
def test_a_yes_or_no_is_read_from_the_words_of_the_next_sentence(
    spoken, sentence, confirmed
):
    speaker = answer_later(spoken, sentence)

    answers = spoken.ask(confirmation())

    speaker.join()
    assert answers == {AnswerKey.CONFIRMED: confirmed}


def test_a_sentence_without_yes_or_no_answers_nothing(spoken):
    speaker = answer_later(spoken, "the kitchen please")

    answers = spoken.ask(confirmation())

    speaker.join()
    assert answers == {}


@pytest.mark.parametrize("filler", ["huh", "hmm", "uh um"])
def test_a_filler_is_no_answer_and_the_sentence_after_it_is(spoken, filler):
    speaker = answer_later(spoken, filler, "yes")

    answers = spoken.ask(confirmation())

    speaker.join()
    assert answers == {AnswerKey.CONFIRMED: True}


def test_an_open_question_is_answered_by_the_sentence_itself(spoken):
    question = Question(
        text="Which cup?",
        person_id="person_abc",
        answers_schema={"object": {"type": "string"}},
    )
    speaker = answer_later(spoken, "the one in the living room")

    answers = spoken.ask(question)

    speaker.join()
    assert answers == {"object": "the one in the living room"}


def test_the_question_is_said_before_listening(spoken):
    speaker = answer_later(spoken, "yes")

    spoken.ask(confirmation())

    speaker.join()
    assert spoken.said == [confirmation().text]


def test_a_sentence_before_the_question_is_no_answer(spoken):
    spoken.on_speech(LiveSpeech(final="yes"))
    spoken.answer_timeout_seconds = 0.1

    with pytest.raises(SkillCallFailed):
        spoken.ask(confirmation())


def test_an_answer_is_reported_as_taken_once(spoken):
    speaker = answer_later(spoken, "yes")
    spoken.ask(confirmation())
    speaker.join()

    assert [spoken.took_as_answer("yes"), spoken.took_as_answer("yes")] == [True, False]


def test_every_tracked_voice_is_listened_to_once(spoken):
    spoken.follow_voices(IdsList(ids=["voice_1", "voice_2"]))
    spoken.follow_voices(IdsList(ids=["voice_1"]))

    assert sorted(spoken.speech_subscriptions) == ["voice_1", "voice_2"]
