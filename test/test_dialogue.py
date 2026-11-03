from __future__ import annotations

import pytest

from speech2plan.contract import (
    EntityConstraint,
    IntentName,
    ParsedInstruction,
    Qualifier,
    Relation,
    RoleBinding,
    RoleConfidence,
    SpatialRelation,
    ThematicRole,
)
from speech2plan.dialogue import Clarifier, RoleQuestion
from speech2plan.grounding import (
    Ambiguity,
    Grounder,
    NoAssumption,
    ResolvedRole,
    Unmatched,
)
from speech2plan.parsing.answer_interpreter import AnswerInterpreter
from speech2plan.placement import QualifierAttribute
from speech2plan.world_source import CareWard, PrototypeWard

from .fake_model import FakeModel
from .fake_skills import FakeDialogueSkills

# %% fixtures


@pytest.fixture(scope="module")
def ward():
    return PrototypeWard().load()


def instruction_for(entity_type: str, role: ThematicRole, **kwargs):
    """A parsed instruction naming one entity type in one role."""
    return ParsedInstruction(
        utterance="take it there",
        speaker="person_abc",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(role=role, constraint=EntityConstraint(entity_type=entity_type))
        ],
        intent_confidence=0.9,
        confidence_per_role=[RoleConfidence(role=role, confidence=0.9)],
        **kwargs,
    )


def clarifier_for(ward, skills, **kwargs) -> Clarifier:
    settings = {"confirm_before_acting": False} | kwargs
    return Clarifier(skills=skills, grounder=grounder_for(ward), **settings)


def grounder_for(ward) -> Grounder:
    """A grounder that assumes nothing, so every ambiguity reaches the dialogue."""
    return Grounder(world=ward, preference=NoAssumption())


# %% questions a person can answer


def test_an_ambiguous_room_is_offered_by_its_floor(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)
    outcomes = grounder_for(ward).ground(instruction)

    questions = clarifier_for(ward, skills).questions_for(instruction, outcomes)

    assert len(questions) == 1
    assert questions[0].role == ThematicRole.GOAL
    assert questions[0].person_id == "person_abc"
    assert questions[0].answers_schema == {
        "goal": {
            "type": "string",
            "enum": ["the one on the ground floor", "the one on the first floor"],
        }
    }
    assert questions[0].text == (
        "Which kitchen do you mean: the one on the ground floor, "
        "or the one on the first floor?"
    )
    assert questions[0].options == {
        "the one on the ground floor": "kitchen_ground",
        "the one on the first floor": "kitchen_first",
    }


def test_an_ambiguous_object_is_offered_by_its_room(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Cup", ThematicRole.OBJECT)
    outcomes = Grounder(world=ward).ground(instruction)

    question = clarifier_for(ward, skills).questions_for(instruction, outcomes)[0]

    assert question.options == {
        "the one in the kitchen on the ground floor": "cup",
        "the one in the living room on the ground floor": "living_room_cup",
    }
    assert "Which cup do you mean" in question.text


def test_the_floor_the_speaker_is_on_settles_a_room_without_asking(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)

    outcomes = Grounder(world=ward).ground(instruction)

    assert isinstance(outcomes[0], ResolvedRole)
    assert outcomes[0].assumed
    assert str(outcomes[0].entity.name) == "kitchen_ground"
    assert clarifier_for(ward, skills).questions_for(instruction, outcomes) == []


def test_namesakes_in_one_room_get_an_open_question(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Plate", ThematicRole.OBJECT)
    outcomes = Grounder(world=ward).ground(instruction)

    question = clarifier_for(ward, skills).questions_for(instruction, outcomes)[0]

    assert question.options == {}
    assert question.answers_schema == {"object": {"type": "string"}}
    assert question.text == (
        "I cannot tell the plates apart in words. Which plate do you mean?"
    )


def test_an_unfilled_role_is_asked_about_in_words(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for(
        "Table", ThematicRole.GOAL, unfilled_roles=[ThematicRole.OBJECT]
    )
    outcomes = Grounder(world=ward).ground(instruction)

    questions = clarifier_for(ward, skills).questions_for(instruction, outcomes)

    assert [question.role for question in questions] == [ThematicRole.OBJECT]
    assert questions[0].text == str(RoleQuestion.OBJECT)


def test_a_missing_entity_is_named_in_words(ward):
    skills = FakeDialogueSkills()
    instruction = ParsedInstruction(
        utterance="go to the kitchen on the first floor",
        speaker="person_abc",
        intent=IntentName.MOVE_TO,
        roles=[
            RoleBinding(
                role=ThematicRole.GOAL,
                constraint=EntityConstraint(
                    entity_type="Bathroom",
                    qualifiers=[
                        Qualifier(
                            attribute=QualifierAttribute.FLOOR, value="GroundFloor"
                        )
                    ],
                ),
            )
        ],
        intent_confidence=0.9,
    )
    outcomes = Grounder(world=ward).ground(instruction)

    question = clarifier_for(ward, skills).questions_for(instruction, outcomes)[0]

    assert question.text == (
        "I cannot find the bathroom on the ground floor. Which one do you mean?"
    )
    assert question.places == []


def test_a_resolved_confident_instruction_asks_nothing(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Table", ThematicRole.GOAL)
    outcomes = Grounder(world=ward).ground(instruction)

    assert clarifier_for(ward, skills).questions_for(instruction, outcomes) == []


def test_a_task_just_stated_is_confirmed_without_stating_it_again(ward):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": True})
    instruction = instruction_for("Table", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills, confirm_before_acting=True).clarify(
        instruction
    )

    assert clarification.confirmed
    assert len(skills.asked) == 1
    assert skills.asked[0].role is None
    assert skills.asked[0].answers_schema == {"confirmed": {"type": "boolean"}}
    assert skills.asked[0].text == "Is that correct?"


# %% verbalisation


def test_verbalising_states_the_task_in_words(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Table", ThematicRole.GOAL)
    outcomes = Grounder(world=ward).ground(instruction)

    clarifier_for(ward, skills).verbalise(instruction, outcomes)

    assert skills.said == [
        "I understood: I will bring something to the table "
        "in the kitchen on the ground floor."
    ]


def test_verbalising_keeps_an_ambiguous_role_indefinite(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)
    outcomes = grounder_for(ward).ground(instruction)

    clarifier_for(ward, skills).verbalise(instruction, outcomes)

    assert skills.said == [
        "I understood: I will bring something to a kitchen. "
        "I know of 2 kitchens: one on the ground floor and one on the first floor."
    ]


def test_verbalising_states_what_was_assumed(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)
    outcomes = Grounder(world=ward).ground(instruction)

    clarifier_for(ward, skills).verbalise(instruction, outcomes)

    assert skills.said == [
        "I understood: I will bring something to the kitchen on the ground floor. "
        "There is also the kitchen on the first floor."
    ]


# %% the answer feeds back into the instruction


def test_the_answer_comes_back_under_the_role(ward):
    skills = FakeDialogueSkills(scripted_answers={"goal": "the one on the first floor"})
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)
    outcomes = grounder_for(ward).ground(instruction)

    answers = clarifier_for(ward, skills).resolve(instruction, outcomes)

    assert len(answers) == 1
    assert answers[0].question.role == ThematicRole.GOAL
    assert answers[0].value == "the one on the first floor"


def test_an_offered_option_pins_the_instance(ward):
    skills = FakeDialogueSkills(scripted_answers={"goal": "the one on the first floor"})
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills).clarify(instruction)

    assert len(skills.asked) == 1
    assert [type(outcome) for outcome in clarification.outcomes] == [ResolvedRole]
    assert str(clarification.outcomes[0].entity.name) == "kitchen_first"
    qualifier = clarification.instruction.constraint_for(ThematicRole.GOAL).qualifiers[
        0
    ]
    assert (qualifier.attribute, qualifier.value) == (
        QualifierAttribute.NAME,
        "kitchen_first",
    )


def test_a_partial_answer_binds_nothing_without_an_interpreter(ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "the one on the ground floor"}
    )
    instruction = instruction_for("Cup", ThematicRole.OBJECT)

    clarification = clarifier_for(ward, skills).clarify(instruction)

    assert isinstance(clarification.outcomes[0], Ambiguity)


def test_an_answer_is_read_against_the_options_of_its_own_question(ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "the one on the ground floor"}
    )
    model = FakeModel(reply='{"choice": "the one in the kitchen on the ground floor"}')
    instruction = instruction_for("Cup", ThematicRole.OBJECT)

    clarification = clarifier_for(
        ward, skills, interpreter=AnswerInterpreter(model=model)
    ).clarify(instruction)

    assert isinstance(clarification.outcomes[0], ResolvedRole)
    assert str(clarification.outcomes[0].entity.name) == "cup"
    assert "the one in the kitchen on the first floor" in model.requests[0].system


def test_an_answer_in_free_words_also_pins_the_instance(ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "the cup in the kitchen on the first floor"}
    )
    instruction = instruction_for("Cup", ThematicRole.OBJECT)

    clarification = clarifier_for(ward, skills).clarify(instruction)

    assert str(clarification.outcomes[0].entity.name) == "second_cup"


def test_an_answered_unfilled_role_becomes_a_binding(ward):
    skills = FakeDialogueSkills(scripted_answers={"goal": "kitchen_first"})
    instruction = instruction_for(
        "Table", ThematicRole.OBJECT, unfilled_roles=[ThematicRole.GOAL]
    )

    clarification = clarifier_for(ward, skills).clarify(instruction)

    assert clarification.instruction.unfilled_roles == []
    assert clarification.instruction.constraint_for(ThematicRole.GOAL).entity_type == (
        "Kitchen"
    )
    resolved = {
        str(outcome.role): str(outcome.entity.name)
        for outcome in clarification.outcomes
        if isinstance(outcome, ResolvedRole)
    }
    assert resolved == {"object": "table", "goal": "kitchen_first"}


def test_a_question_is_asked_only_once(ward):
    skills = FakeDialogueSkills(scripted_answers={"goal": "the garage"})
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills, maximum_rounds=2).clarify(instruction)

    assert len(skills.asked) == 1
    assert isinstance(clarification.outcomes[0], Ambiguity)
    assert clarification.confirmed


def test_a_refused_confirmation_withdraws_the_instruction(ward):
    skills = FakeDialogueSkills(scripted_answers={"confirmed": False})
    instruction = instruction_for("Table", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills, confirm_before_acting=True).clarify(
        instruction
    )

    assert not clarification.confirmed
    assert len(skills.asked) == 1


def test_an_unanswered_confirmation_withdraws_the_instruction(ward):
    skills = FakeDialogueSkills()
    instruction = instruction_for("Table", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills, confirm_before_acting=True).clarify(
        instruction
    )

    assert not clarification.confirmed


def test_the_confirmation_states_what_the_answers_settled(ward):
    skills = FakeDialogueSkills(
        scripted_answers={"goal": "the one on the first floor", "confirmed": True}
    )
    instruction = instruction_for("Kitchen", ThematicRole.GOAL)

    clarification = clarifier_for(ward, skills, confirm_before_acting=True).clarify(
        instruction
    )

    assert clarification.confirmed
    assert skills.asked[-1].text == (
        "I will bring something to the kitchen on the first floor. Is that correct?"
    )


# %% what the twin does not hold


@pytest.fixture(scope="module")
def care_ward():
    return CareWard().load()


def bring(constraint: EntityConstraint) -> ParsedInstruction:
    """Bring what `constraint` describes to the kitchen."""
    return ParsedInstruction(
        utterance="bring it to the kitchen",
        speaker="person_abc",
        intent=IntentName.BRING_OBJECT,
        roles=[
            RoleBinding(role=ThematicRole.OBJECT, constraint=constraint),
            RoleBinding(
                role=ThematicRole.GOAL,
                constraint=EntityConstraint(
                    entity_type="Kitchen",
                    qualifiers=[
                        Qualifier(
                            attribute=QualifierAttribute.FLOOR, value="GroundFloor"
                        )
                    ],
                ),
            ),
        ],
        intent_confidence=0.9,
    )


def gloves():
    return EntityConstraint(entity_type=None, spoken_form="the gloves")


def outcome_for(clarification, role: ThematicRole):
    return next(outcome for outcome in clarification.outcomes if outcome.role == role)


def test_a_thing_the_twin_lacks_is_stated_and_its_place_asked(care_ward):
    skills = FakeDialogueSkills()

    clarifier_for(care_ward, skills).clarify(bring(gloves()))

    assert skills.said == [
        "I understood: I will bring the gloves to the kitchen on the ground floor. "
        "I know of no gloves at all."
    ]
    assert [question.text for question in skills.asked] == ["Where are the gloves?"]
    assert "bedside_table" in skills.asked[0].places
    assert skills.asked[0].candidates == []


def test_the_place_the_speaker_names_travels_with_the_thing(care_ward):
    skills = FakeDialogueSkills(scripted_answers={"object": "on the side table"})

    clarification = clarifier_for(care_ward, skills).clarify(bring(gloves()))

    assert len(skills.asked) == 1
    assert isinstance(outcome_for(clarification, ThematicRole.OBJECT), Unmatched)
    located = clarification.instruction.constraint_for(ThematicRole.OBJECT)
    assert located.spoken_form == "the gloves"
    assert located.relations[0].relation == SpatialRelation.ON
    assert located.relations[0].reference.qualifiers == [
        Qualifier(attribute=QualifierAttribute.NAME, value="bedside_table")
    ]


def test_a_located_thing_is_confirmed_where_it_is(care_ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "on the side table", "confirmed": True}
    )

    clarifier_for(care_ward, skills, confirm_before_acting=True).clarify(
        bring(gloves())
    )

    assert skills.asked[-1].text == (
        "I will bring the gloves on the side table in the bedroom on the ground floor "
        "to the kitchen on the ground floor. Is that correct?"
    )


def test_a_place_that_holds_one_finds_it(care_ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "the living room on the first floor"}
    )
    book_in_the_kitchen = EntityConstraint(
        entity_type="Book",
        relations=[
            Relation(
                relation=SpatialRelation.IN,
                reference=EntityConstraint(entity_type="Kitchen"),
            )
        ],
    )

    clarification = clarifier_for(care_ward, skills).clarify(bring(book_in_the_kitchen))

    assert [question.text for question in skills.asked] == ["Where is the book?"]
    found = outcome_for(clarification, ThematicRole.OBJECT)
    assert isinstance(found, ResolvedRole)
    assert str(found.entity.name) == "lounge_book"


def test_an_answer_may_name_one_the_twin_holds_elsewhere(care_ward):
    skills = FakeDialogueSkills(
        scripted_answers={"object": "the one on the counter top"}
    )
    cup_in_the_bathroom = EntityConstraint(
        entity_type="Cup",
        relations=[
            Relation(
                relation=SpatialRelation.IN,
                reference=EntityConstraint(entity_type="Bathroom"),
            )
        ],
    )

    clarification = clarifier_for(care_ward, skills).clarify(bring(cup_in_the_bathroom))

    assert str(outcome_for(clarification, ThematicRole.OBJECT).entity.name) == (
        "counter_cup"
    )


def test_where_a_thing_is_is_asked_once(care_ward):
    skills = FakeDialogueSkills(scripted_answers={"object": "in the garage"})

    clarification = clarifier_for(care_ward, skills).clarify(bring(gloves()))

    assert len(skills.asked) == 1
    assert clarification.instruction.constraint_for(ThematicRole.OBJECT) == gloves()


def test_an_unnamed_thing_is_offered_by_what_it_is(care_ward):
    skills = FakeDialogueSkills()
    thing_on_the_coffee_table = EntityConstraint(
        entity_type="Thing",
        relations=[
            Relation(
                relation=SpatialRelation.ON,
                reference=EntityConstraint(entity_type="CoffeeTable"),
            )
        ],
    )

    clarifier_for(care_ward, skills).clarify(bring(thing_on_the_coffee_table))

    assert skills.said == [
        "I understood: I will bring a thing on the coffee table to the kitchen on "
        "the ground floor. I know of 2 things like that: the book and the banana."
    ]
    assert skills.asked[0].text == "Which one do you mean: the book, or the banana?"
