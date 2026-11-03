#!/usr/bin/env python3
"""Parse one utterance against the care ward and ground the result.

python3 scripts/parse_once.py "bring my pills from the nightstand"
python3 scripts/parse_once.py --query-world "bring my pills from the nightstand"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
from dataclasses import dataclass, field

from semantic_digital_twin.world import World

from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole
from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser, ReplyKey
from speech2plan.parsing.querying_parser import QueryingInstructionParser, QueryKey
from speech2plan.world_source import CareWard

# %% the parser the arguments ask for


@dataclass
class ShowingQueries(LanguageModel):
    """Passes requests on, printing every query the model puts."""

    model: LanguageModel = field(kw_only=True)

    def complete(self, request: CompletionRequest) -> str:
        reply = self.model.complete(request)
        if f'"{QueryKey.QUERY}"' in reply and f'"{ReplyKey.INTENT}"' not in reply:
            print(f"queried   : {reply.strip()}")
        return reply


def options_for(description: str, default_utterance: str) -> argparse.Namespace:
    """The utterance, and whether the model queries the world."""
    arguments = argparse.ArgumentParser(description=description)
    arguments.add_argument("words", nargs="*", help="the utterance")
    arguments.add_argument(
        "--query-world",
        action="store_true",
        help="name only the types and let the model query the world",
    )
    options = arguments.parse_args()
    options.utterance = " ".join(options.words) or default_utterance
    return options


def parser_for(
    world: World, model: LanguageModel, query_world: bool
) -> InstructionParser:
    """The parser that queries the world, printing its queries, or the one reading a
    listing of it."""
    if query_world:
        return QueryingInstructionParser.for_world(world, ShowingQueries(model=model))
    return InstructionParser.for_world(world, model)


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
    parser = parser_for(world, ClaudeCommandLineModel(), options.query_world)
    instruction = parser.parse(options.utterance)

    print(f"utterance : {instruction.utterance}")
    print(f"intent    : {instruction.intent}")
    for binding in instruction.roles:
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
