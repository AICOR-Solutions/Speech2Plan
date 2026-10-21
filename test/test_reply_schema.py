from __future__ import annotations

import pytest

from speech2plan.contract import IntentName, ThematicRole
from speech2plan.parsing.instruction_parser import InstructionParser, ReplyKey
from speech2plan.vocabulary import EntityOption, Vocabulary
from speech2plan.world_source import ElevatorBuilding

# %% fixtures


@pytest.fixture
def parser() -> InstructionParser:
    vocabulary = Vocabulary(
        intents=list(IntentName),
        entities=[
            EntityOption(entity_type="Kitchen", name="kitchen_ground"),
            EntityOption(entity_type="Bathroom", name="bathroom"),
        ],
    )
    return InstructionParser(model=None, vocabulary=vocabulary)


# %% the world constrains the schema


def test_entity_types_come_from_the_vocabulary(parser):
    schema = parser.reply_schema()
    goal = schema["properties"][str(ReplyKey.ROLES)]["properties"][
        str(ThematicRole.GOAL)
    ]
    assert goal["properties"][str(ReplyKey.ENTITY_TYPE)]["enum"] == [
        "Bathroom",
        "Kitchen",
        None,
    ]


def test_intents_come_from_the_message_vocabulary(parser):
    schema = parser.reply_schema()
    assert schema["properties"][str(ReplyKey.INTENT)]["enum"] == [
        str(intent) for intent in IntentName
    ]


def test_every_role_is_offered(parser):
    schema = parser.reply_schema()
    assert set(schema["properties"][str(ReplyKey.ROLES)]["properties"]) == {
        str(role) for role in ThematicRole
    }


def test_parts_of_other_things_are_not_offered():
    vocabulary = Vocabulary.from_world(ElevatorBuilding(robot_type=None).load())
    assert "Handle" not in vocabulary.entity_types
    assert "Cup" in vocabulary.entity_types
