#!/usr/bin/env python3
"""Turn one utterance into a plan in the elevator building and carry it out.

Parses and clarifies in the terminal as clarify_once.py does, publishes nothing,
and hands the intent it would have published to the plan builder, which plans
it and runs it on a simulated TIAGo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

from clarify_once import ConsoleSkills
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.robots.tiago import Tiago

from speech2plan.contract import ThematicRole
from speech2plan.dialogue import Clarifier
from speech2plan.grounding import Grounder
from speech2plan.intent_message import PublishedIntent
from speech2plan.parsing.answer_interpreter import AnswerInterpreter
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.planning import KinematicSimulation, PlanBuilder
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import ElevatorBuilding

# %% entry point


def main() -> None:
    utterance = " ".join(sys.argv[1:]) or "Bring the cup to the day room"
    world = ElevatorBuilding().load()
    model = ClaudeCommandLineModel()
    instruction = InstructionParser(
        model=model, vocabulary=Vocabulary.from_world(world)
    ).parse(utterance, speaker="person_console")
    clarification = Clarifier(
        skills=ConsoleSkills(),
        grounder=Grounder(world=world),
        interpreter=AnswerInterpreter(model=model),
    ).clarify(instruction)
    if not clarification.confirmed:
        print("withdrawn, nothing to plan")
        return

    message = PublishedIntent(
        instruction=clarification.instruction, outcomes=clarification.outcomes
    ).as_message()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    context = Context(world, robot, alternative_motion_mappings=[TiagoMoveSim])
    steps = PlanBuilder(context=context).steps_for(
        message.intent, json.loads(message.data)
    )
    print("\nplan:")
    for number, step in enumerate(steps, start=1):
        print(f"  {number}. {step.summary}")

    KinematicSimulation(context=context).perform(steps)
    print(f"\ndone, robot at {position_of(robot)}", end="")
    item = json.loads(message.data).get(str(ThematicRole.OBJECT))
    if item is not None:
        print(
            f", {item} at {position_of(world.get_semantic_annotation_by_name(item))}",
            end="",
        )
    print()


def position_of(entity) -> str:
    x, y, z = entity.root.global_pose.position.to_np().flatten()[:3]
    return f"({x:.2f}, {y:.2f}, {z:.2f})"


if __name__ == "__main__":
    main()
