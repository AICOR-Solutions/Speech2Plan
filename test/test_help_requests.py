from __future__ import annotations

import threading
import time

import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from speech2plan_msgs.action import AskForHelp
from std_skills.msg import Result

from speech2plan.dialogue import DialogueSkills
from speech2plan.exceptions import SkillCallFailed
from speech2plan.ros.help_requests import HelpRequests

from .fake_skills import FakeDialogueSkills

# %% stand-ins


class NobodyAnswers(DialogueSkills):
    """A robot that asks, and hears nothing back in time."""

    def say(self, text: str, person_id: str) -> None:
        pass

    def ask(self, question):
        raise SkillCallFailed(skill="spoken answer", reason="no answer in time")


# %% fixtures


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def nodes(ros):
    """The executive's node and the HRI module's node, spun on a thread of their own."""
    executive = rclpy.create_node("executive_asking_for_help")
    hri = rclpy.create_node("hri_module_under_test")
    executor = MultiThreadedExecutor()
    executor.add_node(executive)
    executor.add_node(hri)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield executive, hri
    executor.shutdown()
    spinner.join(timeout=5.0)
    executive.destroy_node()
    hri.destroy_node()


def asked(executive, path: str, text: str = "Is the insulin on my tray?"):
    """The response to one question put through the skill at `path`."""
    client = ActionClient(executive, AskForHelp, path)
    assert client.wait_for_server(timeout_sec=10.0)
    goal = AskForHelp.Goal(problem=AskForHelp.Goal.LOAD_TRAY, text=text)
    handle = completed(client.send_goal_async(goal))
    return completed(handle.get_result_async())


def completed(future, seconds: float = 10.0):
    deadline = time.monotonic() + seconds
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert future.done()
    return future.result()


# %% asking


@pytest.mark.parametrize("answer", [True, False], ids=["yes", "no"])
def test_the_answer_comes_back_as_yes_or_no(nodes, answer):
    executive, hri = nodes
    skills = FakeDialogueSkills(scripted_answers={"confirmed": answer})
    HelpRequests(
        node=hri, skills=skills, dialogue=threading.Lock(), path=f"/help_{answer}"
    )

    response = asked(executive, f"/help_{answer}")

    assert response.result.yes is answer
    assert response.result.result.error_code == Result.ROS_ENOERR
    [question] = skills.asked
    assert question.text == "Is the insulin on my tray?"
    assert question.answers_schema == {"confirmed": {"type": "boolean"}}


def test_a_question_nobody_answers_fails(nodes):
    executive, hri = nodes
    HelpRequests(
        node=hri, skills=NobodyAnswers(), dialogue=threading.Lock(), path="/help_none"
    )

    response = asked(executive, "/help_none")

    assert not response.result.yes
    assert response.result.result.error_code == Result.ROS_EOTHER


def test_no_instruction_dialogue_runs_while_a_question_is_out(nodes):
    executive, hri = nodes
    dialogue = threading.Lock()
    held_while_asking = []

    class Checking(FakeDialogueSkills):
        def ask(self, question):
            held_while_asking.append(dialogue.locked())
            return super().ask(question)

    HelpRequests(
        node=hri,
        skills=Checking(scripted_answers={"confirmed": True}),
        dialogue=dialogue,
        path="/help_locked",
    )

    asked(executive, "/help_locked")

    assert held_while_asking == [True]
