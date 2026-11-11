#!/usr/bin/env python3
"""Score the parser and grounder on the golden utterances, stage by stage.

python3 scripts/evaluate.py                                    # the claude command line
python3 scripts/evaluate.py --openrouter openai/gpt-4.1-mini   # any OpenRouter model
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from speech2plan.evaluation import Evaluation, evaluation_cases, summary
from speech2plan.grounding import Grounder
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.parsing.openrouter import OpenRouterModel
from speech2plan.world_source import CareWard

# %% entry point


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arguments.add_argument(
        "--openrouter", metavar="MODEL", help="an OpenRouter model id instead of claude"
    )
    options = arguments.parse_args()

    model = (
        OpenRouterModel(model=options.openrouter)
        if options.openrouter
        else ClaudeCommandLineModel()
    )
    world = CareWard().load()
    parser = InstructionParser.for_world(world, model)
    evaluation = Evaluation(parser=parser, grounder=Grounder(world=world))
    results = []
    for case in evaluation_cases():
        result = evaluation.score(case)
        results.append(result)
        mark = "ok  " if result.passed_all else "FAIL"
        print(f"{mark} {case.utterance}  ({result.model_calls} calls)", flush=True)
        for failure in result.failures:
            print(f"       {failure}")
    print()
    print(summary(results))
    if isinstance(model, ClaudeCommandLineModel):
        calls = sum(result.model_calls for result in results)
        print(f"prompt tokens per call     {model.input_tokens / max(calls, 1):.0f}")


if __name__ == "__main__":
    main()
