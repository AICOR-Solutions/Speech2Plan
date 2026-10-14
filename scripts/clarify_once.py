#!/usr/bin/env python3
"""Parse one utterance and clarify it in the terminal, standing in for the robot."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from dataclasses import dataclass
from typing import Dict

from parse_once import describe

from speech2plan.dialogue import CONFIRMED, Clarifier, DialogueSkills, Question
from speech2plan.grounding import Grounder
from speech2plan.parsing.answer_interpreter import AnswerInterpreter
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import PrototypeWard

# %% the terminal in place of the communication skills


@dataclass
class ConsoleSkills(DialogueSkills):
    """Prints what the robot would say and reads the answer from stdin."""

    def say(self, text: str, person_id: str) -> None:
        print(f"robot: {text}")

    def ask(self, question: Question) -> Dict[str, object]:
        print(f"robot: {question.text}")
        print(f"       schema {json.dumps(question.answers_schema)}")
        print("you  : ", end="", flush=True)
        answer = sys.stdin.readline().strip()
        if not answer:
            return {}
        if question.answer_key == CONFIRMED:
            return {CONFIRMED: answer.lower() in ("y", "yes", "yeah", "ok", "true")}
        return {question.answer_key: answer}


# %% entry point


def main() -> None:
    utterance = " ".join(sys.argv[1:]) or "Bring the cup to the kitchen"
    world = PrototypeWard().load()
    model = ClaudeCommandLineModel()
    instruction = InstructionParser(
        model=model, vocabulary=Vocabulary.from_world(world)
    ).parse(utterance, speaker="person_console")

    print(f"utterance : {instruction.utterance}")
    print(f"intent    : {instruction.intent}")
    print(f"confidence: {instruction.confidence} (weakest)")
    clarification = Clarifier(
        skills=ConsoleSkills(),
        grounder=Grounder(world=world),
        interpreter=AnswerInterpreter(model=model),
    ).clarify(instruction)
    if not clarification.confirmed:
        print("withdrawn, nothing to publish")
        return
    print("grounding :")
    for outcome in clarification.outcomes:
        print(f"  {describe(outcome)}")
    print(f"unfilled  : {[str(role) for role in clarification.instruction.unfilled_roles]}")


if __name__ == "__main__":
    main()
