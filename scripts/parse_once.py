#!/usr/bin/env python3
"""Parse one utterance against the prototype world and ground the result."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import PrototypeWard

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
    utterance = " ".join(sys.argv[1:]) or "Bring this to the kitchen"
    world = PrototypeWard().load()
    instruction = InstructionParser(
        model=ClaudeCommandLineModel(), vocabulary=Vocabulary.from_world(world)
    ).parse(utterance)

    print(f"utterance : {instruction.utterance}")
    print(f"intent    : {instruction.intent}")
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
