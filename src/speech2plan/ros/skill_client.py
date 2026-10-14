from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Dict

from communication_skills.action import Ask, Say
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_skills.msg import Meta

from speech2plan.dialogue import DialogueSkills, Question
from speech2plan.exceptions import SkillCallFailed

# %% skill paths


class SkillPath:
    """Where the ROS4HRI communication skills live."""

    SAY = "/skill/say"
    ASK = "/skill/ask"


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

    def say(self, text: str, person_id: str) -> None:
        goal = Say.Goal()
        goal.meta = self.meta()
        goal.person_id = person_id
        goal.input = text
        self.send(self.say_client, goal, SkillPath.SAY)

    def ask(self, question: Question) -> Dict[str, object]:
        goal = Ask.Goal()
        goal.meta = self.meta()
        goal.person_id = question.person_id
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
        return self.wait_for(handle.get_result_async(), path).result

    def wait_for(self, future, path: str):
        """Block on a future that the executor completes on another thread."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=self.answer_timeout_seconds):
            future.cancel()
            raise SkillCallFailed(skill=path, reason="no result in time")
        return future.result()
