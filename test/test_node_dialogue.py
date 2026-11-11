from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import List

import pytest
import rclpy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.world_description.geometry import Scale

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.dialogue import Clarifier, DialogueSkills
from speech2plan.exceptions import (
    LanguageModelCallFailed,
    SkillCallFailed,
    UnusableReply,
)
from speech2plan.grounding import Grounder
from speech2plan.ros.intent_node import Speech2PlanNode, SubmissionNotice
from speech2plan.vocabulary import Vocabulary
from speech2plan.world_source import PrototypeWard

from .fake_hri import FakeListener, FakePerson, FakeVoice
from .fake_skills import FakeDialogueSkills

# %% stand-ins


@dataclass
class FakeParser:
    """Returns one canned parse, whatever is said."""

    instruction: ParsedInstruction = field(kw_only=True)

    def parse(self, utterance: str, speaker: str = "") -> ParsedInstruction:
        return replace(self.instruction, utterance=utterance, speaker=speaker)

    def follow(self, world) -> None:
        self.vocabulary = Vocabulary.from_world(world)


@dataclass
class UnusableReplyParser:
    """Fails on the first utterance, the way an unusable model reply does, then parses."""

    failure: Exception = field(kw_only=True)
    instruction: ParsedInstruction = field(kw_only=True)
    failed: bool = field(kw_only=True, default=False)

    def parse(self, utterance: str, speaker: str = "") -> ParsedInstruction:
        if not self.failed:
            self.failed = True
            raise self.failure
        return replace(self.instruction, utterance=utterance, speaker=speaker)


@dataclass
class RecordingParser(FakeParser):
    """Returns one canned parse, and records every utterance it parsed."""

    parsed: List[str] = field(kw_only=True, default_factory=list)

    def parse(self, utterance: str, speaker: str = "") -> ParsedInstruction:
        self.parsed.append(utterance)
        return super().parse(utterance, speaker=speaker)


@dataclass
class SkillsHearingAnInstructionWhileAsking(FakeDialogueSkills):
    """Skills during whose first question somebody gives the node a new instruction."""

    node: object = field(kw_only=True, default=None)
    instruction: str = field(kw_only=True, default="bring the plate to the kitchen")

    def ask(self, question):
        if not self.asked:
            utter(self.node, self.instruction)
        return super().ask(question)


@dataclass
class SkillsThatTookAnAnswer(FakeDialogueSkills):
    """Skills that have just read one sentence as the answer to a question."""

    answer: str = field(kw_only=True, default="yes")

    def took_as_answer(self, utterance: str) -> bool:
        return utterance == self.answer


@dataclass
class BufferedWorldChange:
    """Holds one change another process made, until the node applies it."""

    world: object = field(kw_only=True)
    missed_messages: List[str] = field(
        kw_only=True, default_factory=lambda: ["a table"]
    )

    def apply_missed_messages(self) -> None:
        with self.world.modify_world():
            Table.create_with_new_body_in_world(
                name="delivered_table", world=self.world, scale=Scale(1.0, 1.0, 0.75)
            )
        self.missed_messages.clear()


@dataclass
class RecordedSubmissions:
    """Keeps the intents the node handed to the plan executive."""

    submitted: List[object] = field(kw_only=True, default_factory=list)

    def submit(self, intent) -> None:
        self.submitted.append(intent)


@dataclass
class RefusedSubmissions(RecordedSubmissions):
    """A plan executive that is not there, or does not take the task."""

    reason: str = field(kw_only=True, default="goal rejected")

    def submit(self, intent) -> None:
        raise SkillCallFailed(skill="/skill/carry_out_intent", reason=self.reason)


class BrokenDialogueSkills(DialogueSkills):
    """A robot with no communication skills running."""

    def say(self, text: str, person_id: str) -> None:
        raise SkillCallFailed(skill="/skill/say", reason="no action server")

    def ask(self, question):
        raise SkillCallFailed(skill="/skill/ask", reason="no action server")


# %% fixtures


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


def open_goal() -> ParsedInstruction:
    """Bring the cup, to a place the speaker left open."""
    return ParsedInstruction(
        utterance="",
        speaker="",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(
                role=ThematicRole.OBJECT,
                constraint=EntityConstraint(entity_type=None, spoken_form="the cup"),
            )
        ],
        unfilled_roles=[ThematicRole.GOAL],
        intent_confidence=0.9,
        confidence_per_role=[RoleConfidence(role=ThematicRole.OBJECT, confidence=0.9)],
    )


def doubted_goal() -> ParsedInstruction:
    """Go to the table, which grounds, scored too low to act on."""
    return ParsedInstruction(
        utterance="",
        speaker="",
        intent=IntentName.MOVE_TO,
        roles=[
            RoleBinding(
                role=ThematicRole.GOAL,
                constraint=EntityConstraint(entity_type="Table"),
            )
        ],
        intent_confidence=0.2,
    )


def node_with(ward, skills, instruction=None, listener=None) -> Speech2PlanNode:
    """A node with a stand-in listener, recording the tasks it submits."""
    grounder = Grounder(world=ward)
    node = Speech2PlanNode(
        parser=FakeParser(instruction=instruction or open_goal()),
        grounder=grounder,
        clarifier=(
            None if skills is None else Clarifier(skills=skills, grounder=grounder)
        ),
        listener=listener or FakeListener(),
    )
    node.task_submission = RecordedSubmissions()
    return node


def utter(node, utterance: str, voice_id: str = "voice_1") -> None:
    """Track a voice and let it speak, the way libhri would."""
    node.listener.track(FakeVoice(id=voice_id)).say(utterance)


# %% who spoke


def test_a_voice_bound_to_a_person_is_attributed(ros, ward):
    voice = FakeVoice(id="voice_1")
    listener = FakeListener(
        tracked_persons={"person_abc": FakePerson(id="person_abc", voice=voice)}
    )
    node = node_with(ward, None, listener=listener)

    listener.track(voice).say("bring this to the kitchen")

    submitted = node.task_submission.submitted
    assert len(submitted) == 1
    assert submitted[0].source == "person_abc"
    assert json.loads(submitted[0].data)["locale"] == "en_US"
    node.destroy_node()


def test_a_voice_nobody_claims_is_unknown(ros, ward):
    listener = FakeListener(tracked_persons={"person_abc": FakePerson(id="person_abc")})
    node = node_with(ward, None, listener=listener)

    utter(node, "bring this to the kitchen")

    assert node.task_submission.submitted[0].source == "__unknown_agent__"
    node.destroy_node()


# %% the dialogue in the node's path


def test_the_answer_reaches_the_submitted_intent(ros, ward):
    voice = FakeVoice(id="voice_1")
    listener = FakeListener(
        tracked_persons={"person_abc": FakePerson(id="person_abc", voice=voice)}
    )
    skills = FakeDialogueSkills(
        scripted_answers={"goal": "the kitchen on the first floor", "confirmed": True}
    )
    node = node_with(ward, skills, listener=listener)

    listener.track(voice).say("bring the cup")

    assert len(skills.said) == 1
    assert [question.role for question in skills.asked] == [ThematicRole.GOAL, None]
    assert skills.asked[0].person_id == "person_abc"
    submitted = node.task_submission.submitted
    assert len(submitted) == 1
    data = json.loads(submitted[0].data)
    assert (data["object"], data["goal"]) == ("cup", "kitchen_first")
    node.destroy_node()


def test_without_a_clarifier_an_open_place_stays_a_constraint(ros, ward):
    node = node_with(ward, None)

    utter(node, "bring the cup")

    data = json.loads(node.task_submission.submitted[0].data)
    assert "goal" not in data
    assert data["object"] == "cup"
    [constraint] = data["other_data"]["constraints"]
    assert (constraint["role"], constraint["status"]) == ("goal", "unfilled")
    node.destroy_node()


def test_a_refusal_submits_nothing(ros, ward):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": False})
    node = node_with(ward, skills, doubted_goal())

    utter(node, "do the thing")

    assert node.task_submission.submitted == []
    assert skills.said[0].startswith(
        "I understood: I will go to the table in the kitchen on the ground floor."
    )
    assert skills.asked[-1].text == "Is that correct?"
    node.destroy_node()


def test_a_missing_skill_server_submits_nothing(ros, ward):
    node = node_with(ward, BrokenDialogueSkills())

    utter(node, "bring this to the kitchen")

    assert node.task_submission.submitted == []
    node.destroy_node()


@pytest.mark.parametrize("reason", ["no action server", "goal rejected"])
def test_a_task_the_executive_does_not_take_is_said_to_be_not_taken(
    ros, ward, reason
):
    skills = FakeDialogueSkills(
        scripted_answers={
            "goal": "the kitchen on the first floor",
            "confirmed": True,
        }
    )
    node = node_with(ward, skills)
    node.task_submission = RefusedSubmissions(reason=reason)

    utter(node, "bring the cup here")

    assert skills.said[-1] == SubmissionNotice.NOT_TAKEN
    node.destroy_node()


@pytest.mark.parametrize(
    "failure",
    [
        UnusableReply(utterance="bring me a book", problems=["no type Person"]),
        LanguageModelCallFailed(command="claude", exit_code=1, standard_error=""),
    ],
    ids=["unusable_reply", "no_model_reply"],
)
def test_an_unusable_parse_is_dropped_and_the_next_utterance_is_heard(
    ros, ward, failure
):
    skills = FakeDialogueSkills(
        scripted_answers={
            "goal": "the kitchen on the first floor",
            "confirmed": True,
        }
    )
    node = node_with(ward, skills)
    node.parser = UnusableReplyParser(failure=failure, instruction=open_goal())

    utter(node, "bring me a book")
    utter(node, "bring this to the kitchen")

    assert len(node.task_submission.submitted) == 1
    node.destroy_node()


def test_a_sentence_taken_as_an_answer_is_no_new_instruction(ros, ward):
    skills = SkillsThatTookAnAnswer(
        scripted_answers={
            "goal": "the kitchen on the first floor",
            "confirmed": True,
        }
    )
    node = node_with(ward, skills)

    utter(node, "yes")

    assert (skills.said, node.task_submission.submitted) == ([], [])
    node.destroy_node()


def test_a_sentence_without_a_task_is_dropped_without_a_word(ros, ward):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": True})
    no_task = replace(open_goal(), intent=IntentName.RAW_USER_INPUT, roles=[])
    node = node_with(ward, skills, instruction=no_task)

    utter(node, "huh")

    assert (skills.said, skills.asked, node.task_submission.submitted) == ([], [], [])
    node.destroy_node()


@pytest.mark.parametrize("filler", ["huh", "hmm", "uh um"])
def test_a_filler_is_never_parsed(ros, ward, filler):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": True})
    node = node_with(ward, skills)

    utter(node, filler)

    assert (skills.said, skills.asked, node.task_submission.submitted) == ([], [], [])
    node.destroy_node()


def test_an_instruction_during_a_dialogue_is_not_parsed(ros, ward):
    skills = SkillsHearingAnInstructionWhileAsking(scripted_answers={"confirmed": True})
    node = node_with(ward, skills)
    node.parser = RecordingParser(instruction=open_goal())
    skills.node = node

    utter(node, "bring the cup to the kitchen")

    assert node.parser.parsed == ["bring the cup to the kitchen"]
    node.destroy_node()


# %% the world changing under the node


def test_a_world_change_is_applied_before_the_next_parse(ros):
    world = PrototypeWard().load()
    node = node_with(world, None)
    node.parser.vocabulary = Vocabulary.from_world(world)
    node.world_synchronizer = BufferedWorldChange(world=world)

    types_before = node.parser.vocabulary.entity_types

    utter(node, "bring the cup to the kitchen")

    assert node.parser.vocabulary.entity_types == types_before
    tables = next(
        entry for entry in node.parser.vocabulary.types if entry.entity_type == "Table"
    )
    assert tables.count == 2
    node.destroy_node()
