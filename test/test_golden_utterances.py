from __future__ import annotations

import os

import pytest

from speech2plan.evaluation import Evaluation, evaluation_cases
from speech2plan.grounding import Grounder
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.parsing.querying_parser import QueryingInstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_queries import WorldQueries
from speech2plan.world_source import CareWard

# %% gate


class EnvironmentVariable:
    """Variables that turn this module on."""

    RUN_MODEL_TESTS = "SPEECH2PLAN_RUN_MODEL_TESTS"


pytestmark = pytest.mark.skipif(
    os.environ.get(EnvironmentVariable.RUN_MODEL_TESTS) != "1",
    reason=f"calls a language model; set {EnvironmentVariable.RUN_MODEL_TESTS}=1",
)


# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return CareWard().load()


@pytest.fixture(scope="module", params=["listing", "querying"])
def evaluation(request, ward):
    """The parser that reads a listing of the world, and the one that queries it."""
    model = ClaudeCommandLineModel()
    vocabulary = Vocabulary.from_world(ward)
    parser = (
        InstructionParser(model=model, vocabulary=vocabulary)
        if request.param == "listing"
        else QueryingInstructionParser(
            model=model,
            vocabulary=vocabulary,
            queries=WorldQueries(grounder=Grounder(world=ward)),
        )
    )
    return Evaluation(parser=parser, grounder=Grounder(world=ward))


# %% the eval set


@pytest.mark.parametrize("case", evaluation_cases(), ids=lambda case: case.utterance)
def test_every_stage_matches_the_expectation(case, evaluation):
    if case.needs_query and not isinstance(
        evaluation.parser, QueryingInstructionParser
    ):
        pytest.xfail("only a parser that queries the world can tell what is meant")
    result = evaluation.score(case)
    assert result.passed_all, "; ".join(result.failures)
