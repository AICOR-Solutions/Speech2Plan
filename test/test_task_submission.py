from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List

import pytest
import rclpy
from hri_actions_msgs.msg import Intent
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from speech2plan_msgs.action import CarryOutIntent
from std_skills.msg import Result

from speech2plan.exceptions import SkillCallFailed
from speech2plan.ros.task_submission import TaskSkillPath, TaskSubmission

# %% a plan executive standing in for CRAM's


@dataclass
class StandInExecutive:
    """Serves the task skill: reports one step, then succeeds, or takes no task."""

    node: object = field(kw_only=True)
    takes_tasks: bool = field(kw_only=True, default=True)
    received: List[CarryOutIntent.Goal] = field(kw_only=True, default_factory=list)

    def __post_init__(self):
        self.server = ActionServer(
            self.node,
            CarryOutIntent,
            str(TaskSkillPath.CARRY_OUT_INTENT),
            execute_callback=self.carry_out,
            goal_callback=self.take,
            callback_group=ReentrantCallbackGroup(),
        )

    def take(self, goal) -> GoalResponse:
        self.received.append(goal)
        return GoalResponse.ACCEPT if self.takes_tasks else GoalResponse.REJECT

    def carry_out(self, handle):
        feedback = CarryOutIntent.Feedback(step="drive to kitchen", step_count=1)
        feedback.step_number = 1
        handle.publish_feedback(feedback)
        handle.succeed()
        return CarryOutIntent.Result(result=Result(error_code=Result.ROS_ENOERR))


def bring_intent() -> Intent:
    return Intent(intent=Intent.BRING_OBJECT, data='{"object": "cup"}')


# %% fixtures


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def nodes(ros):
    """A client node and an executive node, spun on a thread of their own."""
    client = rclpy.create_node("task_submission_under_test")
    executive = rclpy.create_node("stand_in_executive")
    executor = MultiThreadedExecutor()
    executor.add_node(client)
    executor.add_node(executive)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield client, executive
    executor.shutdown()
    spinner.join(timeout=5.0)
    client.destroy_node()
    executive.destroy_node()


def wait_until(condition, seconds: float = 5.0) -> bool:
    finished = threading.Event()
    for _ in range(int(seconds / 0.05)):
        if condition():
            return True
        finished.wait(0.05)
    return condition()


# %% submitting


def test_a_task_taken_on_is_carried_out_without_waiting_for_its_end(nodes):
    client, executive = nodes
    StandInExecutive(node=executive)
    submission = TaskSubmission(node=client)

    handle = submission.submit(bring_intent())

    assert handle.accepted
    assert wait_until(lambda: submission.results)
    assert submission.results[0].result.error_code == Result.ROS_ENOERR


def test_the_goal_carries_the_intent_and_who_submitted_it(nodes):
    client, executive = nodes
    stand_in = StandInExecutive(node=executive)

    TaskSubmission(node=client, priority=200).submit(bring_intent())

    [goal] = stand_in.received
    assert goal.intent.intent == Intent.BRING_OBJECT
    assert goal.intent.data == '{"object": "cup"}'
    assert (goal.meta.caller, goal.meta.priority) == ("task_submission_under_test", 200)


def test_a_task_the_executive_does_not_take_fails(nodes):
    client, executive = nodes
    StandInExecutive(node=executive, takes_tasks=False)

    with pytest.raises(SkillCallFailed, match="goal rejected"):
        TaskSubmission(node=client).submit(bring_intent())


def test_without_an_executive_a_task_fails(nodes):
    client, _ = nodes

    with pytest.raises(SkillCallFailed, match="no action server"):
        TaskSubmission(
            node=client, path="/skill/served_by_nobody", timeout_seconds=0.2
        ).submit(bring_intent())
