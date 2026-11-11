from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum

from hri_actions_msgs.msg import Intent
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from speech2plan_msgs.action import AskForHelp
from std_skills.msg import Result

from speech2plan.dialogue import AnswerKey, DialogueSkills, Question
from speech2plan.exceptions import SkillCallFailed

# %% where the plan executive asks


class HelpSkillPath(StrEnum):
    """Where this node takes the questions the plan executive puts to people."""

    ASK_FOR_HELP = "/skill/ask_for_help"


# %% serving


@dataclass
class HelpRequests:
    """Serves the `ask_for_help` skill: says the plan executive's question to the
    people around the robot and hears their yes or no.

    The answer is read from the words alone, without a language model, so the robot
    can ask where it has no network, as inside the elevator. No instruction dialogue
    runs while a question is out, so the answer is not taken for a new instruction.
    """

    node: Node = field(kw_only=True)
    skills: DialogueSkills = field(kw_only=True)
    dialogue: threading.Lock = field(kw_only=True)
    """Held while the node is in a dialogue about an instruction."""

    path: str = field(kw_only=True, default=str(HelpSkillPath.ASK_FOR_HELP))

    def __post_init__(self):
        self.server = ActionServer(
            self.node,
            AskForHelp,
            self.path,
            execute_callback=self.ask,
            callback_group=ReentrantCallbackGroup(),
        )

    def ask(self, goal) -> AskForHelp.Result:
        """Put the question of `goal`, and answer whether the person said yes."""
        request = goal.request
        question = Question(
            text=request.text,
            answers_schema={str(AnswerKey.CONFIRMED): {"type": "boolean"}},
            person_id=request.person_id or Intent.UNKNOWN_AGENT,
        )
        self.node.get_logger().info(f"asking for help ({request.problem}): {request.text}")
        with self.dialogue:
            try:
                answers = self.skills.ask(question)
            except SkillCallFailed as failure:
                self.node.get_logger().info(f"nobody answered: {failure}")
                goal.abort()
                return AskForHelp.Result(
                    result=Result(error_code=Result.ROS_EOTHER, error_msg=str(failure))
                )
        goal.succeed()
        return AskForHelp.Result(
            result=Result(error_code=Result.ROS_ENOERR),
            yes=answers.get(str(AnswerKey.CONFIRMED)) is True,
        )
