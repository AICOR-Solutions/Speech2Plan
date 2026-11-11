#!/usr/bin/env python3
"""Parse one utterance against the care ward and ground the result.

python3 scripts/parse_once.py "bring my pills to the patient room"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse

from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.world_source import CareWard

# %% the arguments


def options_for(description: str, default_utterance: str) -> argparse.Namespace:
    """The utterance."""
    arguments = argparse.ArgumentParser(description=description)
    arguments.add_argument("words", nargs="*", help="the utterance")
    options = arguments.parse_args()
    options.utterance = " ".join(options.words) or default_utterance
    return options


# %% entry point


def describe(outcome) -> str:
    """One line for a grounding outcome."""
    if isinstance(outcome, ResolvedRole):
        return f"{outcome.role:<10} resolved to {outcome.entity.name}"
    if isinstance(outcome, Ambiguity):
        names = ", ".join(str(room.name) for room in outcome.candidates)
        return f"{outcome.role:<10} ambiguous between {names}"
    return f"{outcome.role:<10} matched nothing in the world"


def main() -> None:
    options = options_for(__doc__.splitlines()[0], "Bring this to the kitchen")
    world = CareWard().load()
    parser = InstructionParser.for_world(world, ClaudeCommandLineModel())
    instruction = parser.parse(options.utterance)

    print(f"utterance : {instruction.utterance}")
    print(f"intent    : {instruction.intent}")
    print(f"item      : {instruction.item}")
    for binding in instruction.places:
        print(f"{binding.role:<10}: {binding.constraint.notation()}")
    print(f"unfilled  : {[str(role) for role in instruction.unfilled_roles]}")
    print(f"confidence: {instruction.confidence} (weakest)")
    print(f"  intent    : {instruction.intent_confidence}")
    for scored in instruction.confidence_per_role:
        print(f"  {scored.role:<10}: {scored.confidence}")
    print("grounding :")
    for outcome in Grounder(world=world).ground(instruction):
        print(f"  {describe(outcome)}")


if __name__ == "__main__":
    main()
