#!/usr/bin/env python3
"""Parse one utterance with the Claude command line and print the result.

Uses a hand-built vocabulary so it runs without loading a world.
"""
from __future__ import annotations

import sys

from speech2plan.contract import IntentName, Qualifier, ThematicRole
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import EntityOption, Vocabulary

# %% a stand-in world


def demonstration_vocabulary() -> Vocabulary:
    return Vocabulary(
        intents=list(IntentName),
        entities=[
            EntityOption(
                entity_type="Kitchen",
                name="kitchen_ground",
                qualifiers=[Qualifier(attribute="floor", value="GroundFloor")],
            ),
            EntityOption(
                entity_type="Kitchen",
                name="kitchen_first",
                qualifiers=[Qualifier(attribute="floor", value="FirstFloor")],
            ),
            EntityOption(
                entity_type="LivingRoom",
                name="living_room",
                qualifiers=[Qualifier(attribute="floor", value="GroundFloor")],
            ),
        ],
    )


# %% entry point


def main() -> None:
    utterance = " ".join(sys.argv[1:]) or "Bring this to the kitchen"
    parser = InstructionParser(
        model=ClaudeCommandLineModel(), vocabulary=demonstration_vocabulary()
    )
    instruction = parser.parse(utterance)
    print(f"utterance : {instruction.utterance}")
    print(f"speaker   : {instruction.speaker}")
    print(f"intent    : {instruction.intent}")
    for binding in instruction.roles:
        print(f"  {binding.role:<10}: {binding.constraint}")
    print(f"unfilled  : {[str(role) for role in instruction.unfilled_roles]}")
    print(f"confidence: {instruction.confidence}")


if __name__ == "__main__":
    main()
