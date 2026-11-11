from __future__ import annotations

import pytest

from speech2plan.contract import IntentName, ThematicRole
from speech2plan.parsing.instruction_parser import InstructionParser, ReplyKey
from speech2plan.vocabulary import TypeEntry, Vocabulary

# %% fixtures


@pytest.fixture
def parser() -> InstructionParser:
    vocabulary = Vocabulary(
        intents=list(IntentName),
        types=[TypeEntry(entity_type="Kitchen"), TypeEntry(entity_type="Bathroom")],
        floors=["FirstFloor", "GroundFloor"],
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


def role_schema(parser, role: ThematicRole) -> dict:
    return parser.reply_schema()["properties"][str(ReplyKey.ROLES)]["properties"][
        str(role)
    ]


def goal_schema(parser) -> dict:
    return role_schema(parser, ThematicRole.GOAL)


def test_the_item_is_only_the_speakers_words(parser):
    assert role_schema(parser, ThematicRole.OBJECT)["properties"] == {
        str(ReplyKey.SPOKEN_FORM): {"type": "string"}
    }


def test_a_qualifier_may_only_name_a_floor_of_the_world(parser):
    qualifier = goal_schema(parser)["properties"][str(ReplyKey.QUALIFIERS)]["items"]
    assert qualifier["properties"][str(ReplyKey.ATTRIBUTE)]["enum"] == ["floor"]
    assert qualifier["properties"][str(ReplyKey.VALUE)]["enum"] == [
        "FirstFloor",
        "GroundFloor",
    ]


def test_a_place_lies_in_a_room_that_relates_to_nothing(parser):
    relation = goal_schema(parser)["properties"][str(ReplyKey.RELATIONS)]["items"]
    assert relation["properties"][str(ReplyKey.RELATION)]["enum"] == ["in"]
    reference = relation["properties"][str(ReplyKey.REFERENCE)]
    assert reference["type"] == "object"
    assert str(ReplyKey.RELATIONS) not in reference["properties"]


def test_a_world_without_floors_allows_no_qualifier():
    parser = InstructionParser(
        model=None,
        vocabulary=Vocabulary(
            intents=list(IntentName), types=[TypeEntry(entity_type="Kitchen")]
        ),
    )
    assert goal_schema(parser)["properties"][str(ReplyKey.QUALIFIERS)] == {
        "type": "array",
        "maxItems": 0,
    }
