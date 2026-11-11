#!/usr/bin/env python3
"""Serve /skill/say and /skill/ask on the terminal, in place of the robot's.

Run this next to the node to hear what it says and answer what it asks. With
`--with-executive` it also takes the tasks the node submits on
/skill/carry_out_intent and prints them, in place of CRAM's plan executive.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import rclpy
from clarify_once import typed_answer
from communication_skills.action import Ask, Say
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from speech2plan_msgs.action import CarryOutIntent

from speech2plan.ros.skill_client import SkillPath
from speech2plan.ros.task_submission import TaskSkillPath

# %% the skills, on the terminal


class ConsoleSkills(Node):
    """Prints what the robot would say and reads the answer from stdin."""

    def __init__(self, with_executive: bool = False):
        super().__init__("console_communication_skills")
        group = ReentrantCallbackGroup()
        ActionServer(self, Say, SkillPath.SAY, self.on_say, callback_group=group)
        ActionServer(self, Ask, SkillPath.ASK, self.on_ask, callback_group=group)
        if with_executive:
            ActionServer(
                self,
                CarryOutIntent,
                str(TaskSkillPath.CARRY_OUT_INTENT),
                self.on_task,
                callback_group=group,
            )

    def on_say(self, goal):
        print(f"\nrobot: {goal.request.input}", flush=True)
        goal.succeed()
        return Say.Result()

    def on_ask(self, goal):
        print(f"\nrobot: {goal.request.question}", flush=True)
        requested = json.loads(goal.request.answers_schema)
        for key, schema in requested.items():
            print(f"       {key} {json.dumps(schema)}", flush=True)
        print("you  : ", end="", flush=True)
        answers = typed_answer(requested, sys.stdin.readline().strip())
        goal.succeed()
        result = Ask.Result()
        result.answers = json.dumps(answers)
        return result

    def on_task(self, goal):
        intent = goal.request.intent
        print(f"\ntask : {intent.intent} {intent.data}", flush=True)
        goal.succeed()
        return CarryOutIntent.Result()


# %% entry point


def main() -> None:
    rclpy.init()
    with_executive = "--with-executive" in sys.argv[1:]
    node = ConsoleSkills(with_executive=with_executive)
    served = "/skill/say and /skill/ask"
    if with_executive:
        served += f" and {TaskSkillPath.CARRY_OUT_INTENT}"
    print(f"serving {served}", flush=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
