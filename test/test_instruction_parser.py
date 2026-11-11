from __future__ import annotations

import json

import pytest

from speech2plan.contract import IntentName, SpatialRelation, ThematicRole
from speech2plan.exceptions import UnusableReply
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import CareWard

from .fake_model import FakeModel

# %% fixtures


@pytest.fixture(scope="module")
def vocabulary():
    return Vocabulary.from_world(CareWard().load())


def reply(**roles) -> str:
    """A reply bringing whatever `roles` describe."""
    return json.dumps(
        {
            "intent": str(IntentName.BRING_OBJECT),
            "roles": roles,
            "unfilled_roles": [],
            "confidence": {"intent": 0.9},
        }
    )


def table_in_the_kitchen():
    return {
        "entity_type": "Table",
        "relations": [
            {
                "relation": "in",
                "reference": {"entity_type": "Kitchen", "spoken_form": "the kitchen"},
            }
        ],
        "spoken_form": "the table in the kitchen",
    }


# %% a usable reply


def test_a_relation_becomes_a_described_reference(vocabulary):
    parser = InstructionParser(
        model=FakeModel(reply=reply(goal=table_in_the_kitchen())),
        vocabulary=vocabulary,
    )
    constraint = parser.parse("go to the table in the kitchen").constraint_for(
        ThematicRole.GOAL
    )
    assert constraint.notation() == "Table in Kitchen"
    assert constraint.relations[0].relation == SpatialRelation.IN


def test_a_place_of_no_listed_type_keeps_the_speakers_words(vocabulary):
    pharmacy = {"entity_type": None, "spoken_form": "the pharmacy"}
    parser = InstructionParser(
        model=FakeModel(reply=reply(goal=pharmacy)), vocabulary=vocabulary
    )
    constraint = parser.parse("go to the pharmacy").constraint_for(ThematicRole.GOAL)
    assert constraint.entity_type is None
    assert constraint.spoken_form == "the pharmacy"


def test_the_item_is_kept_in_the_speakers_words(vocabulary):
    parser = InstructionParser(
        model=FakeModel(
            reply=reply(object={"spoken_form": "the insulin"}, goal=table_in_the_kitchen())
        ),
        vocabulary=vocabulary,
    )
    instruction = parser.parse("bring the insulin to the table in the kitchen")
    assert instruction.item == "the insulin"
    assert instruction.constraint_for(ThematicRole.OBJECT).entity_type is None


def test_an_item_described_by_a_type_is_returned(vocabulary):
    parser = InstructionParser(model=None, vocabulary=vocabulary)
    problems = parser.problems_with(
        json.loads(reply(object={"entity_type": "Cup", "spoken_form": "the cup"}))
    )
    assert problems == [
        "roles.object takes only spoken_form: the robot does not look for what it "
        "carries, people put it on the tray"
    ]


def test_the_utterance_is_marked_off_and_the_world_named_without_instances(
    vocabulary,
):
    model = FakeModel(reply=reply(goal=table_in_the_kitchen()))
    InstructionParser(model=model, vocabulary=vocabulary).parse("go to the table")
    request = model.requests[0]
    assert request.user == "<utterance>go to the table</utterance>"
    assert "- Table (2):" in request.system
    assert "first_floor_table" not in request.system
    assert "- Cup" not in request.system


# %% a reply that breaks the rules goes back once


def test_a_type_the_world_lacks_is_returned_with_what_was_wrong(vocabulary):
    model = FakeModel(
        replies=[
            reply(goal={"entity_type": "Garage"}),
            reply(goal=table_in_the_kitchen()),
        ]
    )
    instruction = InstructionParser(model=model, vocabulary=vocabulary).parse(
        "go to the table"
    )

    assert instruction.constraint_for(ThematicRole.GOAL).entity_type == "Table"
    correction = model.requests[1].user
    assert correction.startswith("<utterance>go to the table</utterance>")
    assert "<rejected>" in correction
    assert "'Garage' is not in the world" in correction
    assert '"Garage"' in correction


def test_a_room_named_as_a_qualifier_is_turned_into_a_relation(vocabulary):
    in_the_kitchen = {
        "entity_type": "Table",
        "qualifiers": [{"attribute": "room", "value": "kitchen_ground"}],
    }
    model = FakeModel(
        replies=[reply(goal=in_the_kitchen), reply(goal=table_in_the_kitchen())]
    )
    InstructionParser(model=model, vocabulary=vocabulary).parse("go to the table")
    assert "reference of an 'in' relation" in model.requests[1].user


def test_a_floor_the_world_lacks_is_returned(vocabulary):
    parser = InstructionParser(model=None, vocabulary=vocabulary)
    problems = parser.problems_with(
        json.loads(
            reply(
                goal={
                    "entity_type": "Kitchen",
                    "qualifiers": [{"attribute": "floor", "value": "Basement"}],
                }
            )
        )
    )
    assert problems == [
        "roles.goal names floor 'Basement'; the floors are ['FirstFloor', 'GroundFloor']"
    ]


def test_a_reply_that_is_no_json_is_returned(vocabulary):
    model = FakeModel(replies=["Sure! Here it is.", reply(goal=table_in_the_kitchen())])
    InstructionParser(model=model, vocabulary=vocabulary).parse("go to the table")
    assert "the reply is no JSON object" in model.requests[1].user


def test_a_reply_wrong_every_time_is_unusable(vocabulary):
    model = FakeModel(reply=reply(goal={"entity_type": "Garage"}))
    with pytest.raises(UnusableReply):
        InstructionParser(model=model, vocabulary=vocabulary).parse("go to the garage")
    assert len(model.requests) == 2


@pytest.mark.parametrize(
    "broken",
    [
        {"goal": {"entity_type": "Table", "qualifiers": ["upstairs"]}},
        {"goal": {"entity_type": "Table", "relations": ["in the kitchen"]}},
        {"goal": {"entity_type": "Table", "qualifiers": "floor=FirstFloor"}},
        {"goal": "Table"},
        {"object": "insulin"},
    ],
    ids=["qualifier", "relation", "qualifier_list", "role", "item"],
)
def test_a_malformed_reply_is_a_problem_not_a_crash(vocabulary, broken):
    parser = InstructionParser(model=None, vocabulary=vocabulary)
    assert parser.problems_with(json.loads(reply(**broken)))


def test_a_confidence_for_no_role_is_a_problem(vocabulary):
    parser = InstructionParser(model=None, vocabulary=vocabulary)
    parsed = json.loads(reply(goal=table_in_the_kitchen()))
    parsed["confidence"]["location"] = 0.5
    assert parser.problems_with(parsed) == [
        "confidence must map 'intent' and the filled roles to numbers"
    ]


def test_a_room_relating_in_turn_is_returned(vocabulary):
    bedroom_in_a_kitchen = {
        "entity_type": "SideTable",
        "relations": [
            {
                "relation": "in",
                "reference": {
                    "entity_type": "Bedroom",
                    "relations": [
                        {"relation": "in", "reference": {"entity_type": "Kitchen"}}
                    ],
                },
            }
        ],
    }
    problems = InstructionParser(model=None, vocabulary=vocabulary).problems_with(
        json.loads(reply(goal=bedroom_in_a_kitchen))
    )
    assert problems == [
        "roles.goal.relations[0].reference relates a room; only places lie in rooms"
    ]
