from __future__ import annotations

import os

import pytest

from speech2plan.evaluation import Evaluation, evaluation_cases
from speech2plan.grounding import Grounder
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
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


@pytest.fixture(scope="module")
def evaluation(ward):
    parser = InstructionParser(
        model=ClaudeCommandLineModel(), vocabulary=Vocabulary.from_world(ward)
    )
    return Evaluation(parser=parser, grounder=Grounder(world=ward))


# %% the eval set


@pytest.mark.parametrize("case", evaluation_cases(), ids=lambda case: case.utterance)
def test_every_stage_matches_the_expectation(case, evaluation):
    result = evaluation.score(case)
    assert result.passed_all, "; ".join(result.failures)
