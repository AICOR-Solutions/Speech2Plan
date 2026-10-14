#!/usr/bin/env python3
"""Serve /skill/say and /skill/ask on the terminal, in place of the robot's.

Run this next to the node to hear what it says and answer what it asks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import rclpy
from communication_skills.action import Ask, Say
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

# %% the skills, on the terminal


class ConsoleSkills(Node):
    """Prints what the robot would say and reads the answer from stdin."""

    def __init__(self):
        super().__init__("console_communication_skills")
        group = ReentrantCallbackGroup()
        ActionServer(self, Say, "/skill/say", self.on_say, callback_group=group)
        ActionServer(self, Ask, "/skill/ask", self.on_ask, callback_group=group)

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
        spoken = sys.stdin.readline().strip()
        goal.succeed()
        result = Ask.Result()
        result.answers = json.dumps(self.answers(requested, spoken))
        return result

    def answers(self, requested: dict, spoken: str) -> dict:
        """What the person said, under the key the question asked for."""
        if not spoken:
            return {}
        key = next(iter(requested))
        if requested[key].get("type") == "boolean":
            return {key: spoken.lower() in ("y", "yes", "yeah", "ok", "true")}
        return {key: spoken}


# %% entry point


def main() -> None:
    rclpy.init()
    node = ConsoleSkills()
    print("serving /skill/say and /skill/ask", flush=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
