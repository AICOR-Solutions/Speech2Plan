#!/usr/bin/env python3
"""Parse one utterance and clarify it in the terminal, standing in for the robot."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from dataclasses import dataclass
from typing import Dict

from parse_once import describe, options_for

from speech2plan.dialogue import Clarifier, DialogueSkills, Question
from speech2plan.grounding import Grounder
from speech2plan.parsing.answer_interpreter import AnswerInterpreter
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.world_source import CareWard

# %% the terminal in place of the communication skills


def typed_answer(answers_schema: Dict[str, dict], typed: str) -> Dict[str, object]:
    """What was typed, under the key the question asked for, a yes as true."""
    if not typed:
        return {}
    key = next(iter(answers_schema))
    if answers_schema[key].get("type") == "boolean":
        return {key: typed.lower() in ("y", "yes", "yeah", "ok", "true")}
    return {key: typed}


@dataclass
class ConsoleSkills(DialogueSkills):
    """Prints what the robot would say and reads the answer from stdin."""

    def say(self, text: str, person_id: str) -> None:
        print(f"robot: {text}")

    def ask(self, question: Question) -> Dict[str, object]:
        print(f"robot: {question.text}")
        print(f"       schema {json.dumps(question.answers_schema)}")
        print("you  : ", end="", flush=True)
        return typed_answer(question.answers_schema, sys.stdin.readline().strip())


# %% entry point


def main() -> None:
    options = options_for(__doc__.splitlines()[0], "Bring the insulin to the kitchen")
    world = CareWard().load()
    model = ClaudeCommandLineModel()
    instruction = InstructionParser.for_world(world, model).parse(
        options.utterance, speaker="person_console"
    )

    print(f"utterance : {instruction.utterance}")
    print(f"intent    : {instruction.intent}")
    print(f"confidence: {instruction.confidence} (weakest)")
    clarification = Clarifier(
        skills=ConsoleSkills(),
        grounder=Grounder(world=world),
        interpreter=AnswerInterpreter(model=model),
    ).clarify(instruction)
    if not clarification.confirmed:
        print("withdrawn, nothing to submit")
        return
    print(f"item      : {clarification.instruction.item}")
    print("grounding :")
    for outcome in clarification.outcomes:
        print(f"  {describe(outcome)}")
    print(
        f"unfilled  : {[str(role) for role in clarification.instruction.unfilled_roles]}"
    )


if __name__ == "__main__":
    main()
