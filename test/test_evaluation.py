from __future__ import annotations

import json

import pytest

from speech2plan.contract import IntentName, ThematicRole
from speech2plan.evaluation import (
    Evaluation,
    EvaluationCase,
    ExpectedGrounding,
    Stage,
    summary,
)
from speech2plan.grounding import Grounder
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import CareWard

from .fake_model import FakeModel

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return CareWard().load()


def evaluation_answering(ward, reply: dict) -> Evaluation:
    return Evaluation(
        parser=InstructionParser(
            model=FakeModel(reply=json.dumps(reply)),
            vocabulary=Vocabulary.from_world(ward),
        ),
        grounder=Grounder(world=ward),
    )


@pytest.fixture
def case():
    return EvaluationCase(
        utterance="bring the insulin to the table upstairs",
        intent=IntentName.BRING_OBJECT,
        item="insulin",
        entity_type_per_role={ThematicRole.GOAL: "Table"},
        grounding_per_role={
            ThematicRole.GOAL: ExpectedGrounding.resolved("first_floor_table")
        },
    )


UPSTAIRS = [{"attribute": "floor", "value": "FirstFloor"}]


def reply(
    intent=IntentName.BRING_OBJECT, item="the insulin", goal_qualifiers=UPSTAIRS
) -> dict:
    return {
        "intent": str(intent),
        "roles": {
            "object": {"spoken_form": item},
            "goal": {"entity_type": "Table", "qualifiers": goal_qualifiers},
        },
        "confidence": {"intent": 0.9},
    }


# %% scoring


def test_a_right_parse_passes_every_stage(ward, case):
    result = evaluation_answering(ward, reply()).score(case)
    assert result.passed_all, result.failures


def test_a_wrong_intent_fails_only_that_stage(ward, case):
    result = evaluation_answering(ward, reply(intent=IntentName.MOVE_TO)).score(
        case
    )
    assert [stage for stage, passed in result.passed.items() if not passed] == [
        Stage.INTENT
    ]


def test_a_wrong_item_fails_only_the_roles(ward, case):
    result = evaluation_answering(ward, reply(item="the gloves")).score(case)
    assert [stage for stage, passed in result.passed.items() if not passed] == [
        Stage.ROLES
    ]


def test_a_place_left_out_fails_the_grounding(ward, case):
    result = evaluation_answering(ward, reply(goal_qualifiers=[])).score(case)
    assert not result.passed[Stage.GROUNDING]
    assert result.passed[Stage.ROLES]
    assert result.failures == [
        f"grounding goal={ExpectedGrounding.resolved('table')}",
    ]


def test_an_unusable_reply_fails_every_stage(ward, case):
    result = evaluation_answering(ward, {"intent": "dance"}).score(case)
    assert not any(result.passed.values())


def test_the_summary_counts_each_stage(ward, case):
    results = [
        evaluation_answering(ward, reply()).score(case),
        evaluation_answering(ward, reply(intent=IntentName.MOVE_TO)).score(case),
    ]
    table = summary(results).splitlines()
    assert table[2].split() == ["intent", "1/2", "50%"]
    assert table[-3].split() == ["all", "stages", "1/2", "50%"]
    assert table[-2].split() == ["model", "calls", "per", "case", "1.0"]


def test_what_a_case_costs_is_counted(ward, case):
    result = evaluation_answering(ward, reply()).score(case)
    assert result.model_calls == 1
    assert result.prompt_characters > 1000
