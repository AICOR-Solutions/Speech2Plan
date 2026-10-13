from __future__ import annotations

from hri_actions_msgs.msg import Intent

from speech2plan.contract import IntentName, ThematicRole

# %% the vocabulary is the message's, not a copy


def test_intent_names_match_the_ros4hri_message():
    assert IntentName.BRING_OBJECT == Intent.BRING_OBJECT
    assert IntentName.MOVE_TO == Intent.MOVE_TO
    assert IntentName.GUIDE == Intent.GUIDE
    assert IntentName.PLACE_OBJECT == Intent.PLACE_OBJECT
    assert IntentName.RAW_USER_INPUT == Intent.RAW_USER_INPUT


def test_thematic_roles_are_the_documented_keys():
    assert ThematicRole.OBJECT == "object"
    assert ThematicRole.GOAL == "goal"
    assert ThematicRole.RECIPIENT == "recipient"
