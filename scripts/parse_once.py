#!/usr/bin/env python3
"""Parse one utterance against the care ward and ground the result.

python3 scripts/parse_once.py "bring my pills from the nightstand"
python3 scripts/parse_once.py --query-world "bring my pills from the nightstand"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from dataclasses import dataclass, field
from typing import List

from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole
from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.parsing.querying_parser import QueryingInstructionParser, QueryKey
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_queries import WorldQueries
from speech2plan.world_source import CareWard

QUERY_WORLD = "--query-world"
"""The argument that lets the model query the world instead of reading a listing."""


# %% the parser the arguments ask for


@dataclass
class ShowingQueries(LanguageModel):
    """Passes requests on, printing every query the model puts."""

    model: LanguageModel = field(kw_only=True)

    def complete(self, request: CompletionRequest) -> str:
        reply = self.model.complete(request)
        if f'"{QueryKey.QUERY}"' in reply and '"intent"' not in reply:
            print(f"queried   : {reply.strip()}")
        return reply


def utterance_in(arguments: List[str], default: str) -> str:
    """The words among the arguments, the flags left out."""
    return " ".join(word for word in arguments if word != QUERY_WORLD) or default


def parser_for(world, model: LanguageModel, arguments: List[str]) -> InstructionParser:
    """The parser that queries the world when asked to, else the one reading a
    listing of it."""
    if QUERY_WORLD not in arguments:
        return InstructionParser(model=model, vocabulary=Vocabulary.from_world(world))
    return QueryingInstructionParser(
        model=ShowingQueries(model=model),
        vocabulary=Vocabulary.from_world(world, describe_instances=False),
        queries=WorldQueries(grounder=Grounder(world=world)),
    )


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
    utterance = utterance_in(sys.argv[1:], "Bring this to the kitchen")
    world = CareWard().load()
    instruction = parser_for(world, ClaudeCommandLineModel(), sys.argv[1:]).parse(
        utterance
    )

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
