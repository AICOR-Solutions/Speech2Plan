from __future__ import annotations

import os

import pytest

from speech2plan.parsing.claude_cli import ClaudeCommandLineModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import PrototypeWard

from .golden_utterances import golden_utterances

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
    return PrototypeWard().load()


@pytest.fixture(scope="module")
def parser(ward):
    return InstructionParser(
        model=ClaudeCommandLineModel(), vocabulary=Vocabulary.from_world(ward)
    )


# %% the eval set


@pytest.mark.parametrize(
    "golden", golden_utterances(), ids=lambda golden: golden.utterance
)
def test_the_parse_matches_the_expectation(golden, parser):
    instruction = parser.parse(golden.utterance)
    assert instruction.intent == golden.intent
    assert {
        binding.role: binding.constraint.entity_type for binding in instruction.roles
    } == golden.entity_type_per_role
    assert set(instruction.unfilled_roles) == set(golden.unfilled_roles)
