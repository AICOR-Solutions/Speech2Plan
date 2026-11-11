from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List

from hri_actions_msgs.msg import Intent
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from speech2plan_msgs.action import CarryOutIntent
from std_skills.msg import Meta

from speech2plan.exceptions import SkillCallFailed
from speech2plan.ros.skill_client import completes

# %% where tasks go


class TaskSkillPath(StrEnum):
    """Where the plan executive takes the tasks it carries out."""

    CARRY_OUT_INTENT = "/skill/carry_out_intent"


# %% submitting


@dataclass
class TaskSubmission:
    """Hands a confirmed intent to the plan executive as a `CarryOutIntent` goal.

    Submitting waits only until the executive accepted the task, so the node hears the
    next instruction while the robot carries the task out. What the executive reports
    while and after it carries a task out is logged.
    """

    node: Node = field(kw_only=True)
    path: str = field(kw_only=True, default=str(TaskSkillPath.CARRY_OUT_INTENT))
    """Where the plan executive serves the task skill."""

    priority: int = field(kw_only=True, default=Meta.NORMAL_PRIORITY)
    timeout_seconds: float = field(kw_only=True, default=10.0)
    """How long to wait for the executive to be there, and to accept a task."""

    results: List[CarryOutIntent.Result] = field(init=False, default_factory=list)
    """The results of the tasks carried out, in the order they ended."""

    def __post_init__(self):
        self.client = ActionClient(
            self.node,
            CarryOutIntent,
            self.path,
            callback_group=ReentrantCallbackGroup(),
        )

    def submit(self, intent: Intent):
        """Hand `intent` to the plan executive.

        :return: The handle of the accepted goal.
        :raises SkillCallFailed: If no executive is there, or it does not take the task.
        """
        if not self.client.wait_for_server(timeout_sec=self.timeout_seconds):
            raise SkillCallFailed(skill=self.path, reason="no action server")
        goal = CarryOutIntent.Goal()
        goal.meta.caller = self.node.get_name()
        goal.meta.priority = self.priority
        goal.intent = intent
        sent = self.client.send_goal_async(goal, feedback_callback=self.on_feedback)
        if not completes(sent, self.timeout_seconds):
            sent.cancel()
            raise SkillCallFailed(skill=self.path, reason="no answer in time")
        handle = sent.result()
        if not handle.accepted:
            raise SkillCallFailed(skill=self.path, reason="goal rejected")
        handle.get_result_async().add_done_callback(self.on_result)
        return handle

    def on_feedback(self, message) -> None:
        feedback = message.feedback
        doing = {
            feedback.WAITING_FOR_PEOPLE: "waiting for people",
            feedback.ASKING_FOR_HELP: f"asking for help: {feedback.problem}",
        }.get(feedback.state, "running")
        self.node.get_logger().info(
            f"step {feedback.step_number}/{feedback.step_count}: {feedback.step} "
            f"({doing})"
        )

    def on_result(self, future: Future) -> None:
        result = future.result().result
        self.results.append(result)
        outcome = result.result
        for record in result.records:
            self.node.get_logger().info(record)
        if outcome.error_code == outcome.ROS_ENOERR:
            self.node.get_logger().info("task carried out")
        else:
            self.node.get_logger().error(
                f"task not carried out ({outcome.error_code}): {outcome.error_msg}"
            )
