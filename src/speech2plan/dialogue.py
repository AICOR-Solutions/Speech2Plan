from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Dict, List, Optional, Set

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import (
    EntityConstraint,
    ParsedInstruction,
    Qualifier,
    RoleBinding,
    RoleConfidence,
    ThematicRole,
)
from speech2plan.belief import BeliefReport
from speech2plan.description import (
    EntityDescriber,
    definite,
    indefinite,
)
from speech2plan.grounding import (
    Ambiguity,
    Grounder,
    GroundingOutcome,
    ResolvedRole,
    Unmatched,
)
from speech2plan.parsing.answer_interpreter import AnswerInterpreter
from speech2plan.placement import QualifierAttribute

# %% how the robot talks about a task


class AnswerKey(StrEnum):
    """The answer key of a question that fills no role."""

    CONFIRMED = "confirmed"
    """The speaker agrees to the task as stated."""


class SpokenTask(StrEnum):
    """What each intent sounds like, with a placeholder per thematic role."""

    RAW_USER_INPUT = "do what you asked for"
    MOVE_TO = "go to {goal}"
    BRING_OBJECT = "bring {object} to {goal}"


class OpenRole(StrEnum):
    """How to say a role nobody has pinned down yet."""

    AGENT = "someone"
    OBJECT = "something"
    GOAL = "somewhere"
    RECIPIENT = "someone"
    INPUT = "something"
    LOCALE = "some language"


class RoleQuestion(StrEnum):
    """What to ask for a role the speaker left open."""

    AGENT = "Who should do that?"
    OBJECT = "What should I take?"
    GOAL = "Where should I take it?"
    RECIPIENT = "Who should get it?"
    INPUT = "What should I say?"
    LOCALE = "Which language should I use?"


# %% what the robot needs to say or ask


@dataclass
class Question:
    """One question to put to a person, and the answers it will accept."""

    text: str = field(kw_only=True)
    answers_schema: Dict[str, dict] = field(kw_only=True)
    person_id: str = field(kw_only=True)
    role: Optional[ThematicRole] = field(kw_only=True, default=None)
    """The role the answer fills, or None when it only confirms the parse."""

    options: Dict[str, str] = field(kw_only=True, default_factory=dict)
    """The entity behind each offered option, by the words it was offered in."""

    candidates: List[str] = field(kw_only=True, default_factory=list)
    """Every entity an answer may name to fill the role; none for the item, which an
    answer names in words."""

    topic: Optional[str] = field(kw_only=True, default=None)
    """What the question is about, so it is put once however its words change.
    None when its words are what it is about."""

    @property
    def answer_key(self) -> str:
        """The key the answer arrives under."""
        return str(self.role) if self.role else str(AnswerKey.CONFIRMED)


@dataclass
class Answer:
    """What came back for one question."""

    question: Question = field(kw_only=True)
    value: object = field(kw_only=True)


@dataclass
class Clarification:
    """An instruction after the dialogue, and how it grounds now."""

    instruction: ParsedInstruction = field(kw_only=True)
    outcomes: List[GroundingOutcome] = field(kw_only=True)
    confirmed: bool = field(kw_only=True, default=True)


# %% the skills this node calls


class DialogueSkills(ABC):
    """The ROS4HRI communication skills this node uses."""

    @abstractmethod
    def say(self, text: str, person_id: str) -> None:
        """Utter `text` to a person."""

    @abstractmethod
    def ask(self, question: Question) -> Dict[str, object]:
        """Put a question and return the answers that came back."""

    def took_as_answer(self, utterance: str) -> bool:
        """Whether `utterance` already answered a question, and is no instruction."""
        return False


# %% turning grounding outcomes into questions


@dataclass
class Clarifier:
    """Asks what an instruction leaves open, and re-grounds with the answers."""

    skills: DialogueSkills = field(kw_only=True)
    grounder: Grounder = field(kw_only=True)
    describer: Optional[EntityDescriber] = field(kw_only=True, default=None)
    belief: Optional[BeliefReport] = field(kw_only=True, default=None)
    interpreter: Optional[AnswerInterpreter] = field(kw_only=True, default=None)
    """Reads an answer that matches none of the offered ones literally."""

    confirm_before_acting: bool = field(kw_only=True, default=True)
    maximum_rounds: int = field(kw_only=True, default=2)

    def __post_init__(self):
        if self.describer is None:
            self.describer = EntityDescriber(world=self.grounder.world)
        if self.belief is None:
            self.belief = BeliefReport(grounder=self.grounder, describer=self.describer)

    def clarify(self, instruction: ParsedInstruction) -> Clarification:
        """Verbalise the parse, settle what is open, and confirm before acting."""
        outcomes = self.grounder.ground(instruction)
        self.verbalise(instruction, outcomes)
        stated_task = self.describe_task(instruction, outcomes)
        asked = set()
        for _ in range(self.maximum_rounds):
            answers = self.resolve(instruction, outcomes, asked=asked)
            if not answers:
                break
            instruction = self.apply(instruction, answers)
            outcomes = self.grounder.ground(instruction)
        return Clarification(
            instruction=instruction,
            outcomes=outcomes,
            confirmed=self.confirmed(instruction, outcomes, stated_task=stated_task),
        )

    def confirmed(
        self,
        instruction: ParsedInstruction,
        outcomes: List[GroundingOutcome],
        stated_task: str = "",
    ) -> bool:
        """
        Whether the speaker agreed to the task as it now stands, which is stated again
        only if it differs from `stated_task`.
        """
        if not self.confirm_before_acting:
            return True
        task = self.describe_task(instruction, outcomes)
        question = Question(
            text=(
                "Is that correct?"
                if task == stated_task
                else f"I will {task}. Is that correct?"
            ),
            answers_schema={str(AnswerKey.CONFIRMED): {"type": "boolean"}},
            person_id=instruction.speaker,
        )
        return self.skills.ask(question).get(AnswerKey.CONFIRMED) is True

    def verbalise(
        self, instruction: ParsedInstruction, outcomes: List[GroundingOutcome]
    ) -> str:
        """State the task and what the twin holds about it, and say it."""
        text = " ".join(
            [f"I understood: I will {self.describe_task(instruction, outcomes)}."]
            + self.belief.sentences(instruction, outcomes)
        )
        self.skills.say(text, instruction.speaker)
        return text

    def describe_task(
        self, instruction: ParsedInstruction, outcomes: List[GroundingOutcome]
    ) -> str:
        """The whole instruction as one phrase, as far as it is known."""
        words = {str(role): str(OpenRole[role.name]) for role in ThematicRole}
        for binding in instruction.roles:
            spoken = self.describer.constraint_words(binding.constraint)
            words[str(binding.role)] = (
                definite(spoken)
                if binding.constraint.entity_type is None
                else indefinite(spoken)
            )
        for outcome in outcomes:
            if isinstance(outcome, ResolvedRole):
                words[str(outcome.role)] = self.describer.describe(
                    outcome.entity
                ).phrase
        return str(SpokenTask[instruction.intent.name]).format(**words)

    def questions_for(
        self, instruction: ParsedInstruction, outcomes: List[GroundingOutcome]
    ) -> List[Question]:
        """Every question the instruction still needs answered."""
        questions = [
            self.question_for_ambiguity(outcome, instruction.speaker)
            for outcome in outcomes
            if isinstance(outcome, Ambiguity)
        ]
        questions.extend(
            question
            for question in (
                self.question_for_unmatched(outcome, instruction.speaker)
                for outcome in outcomes
                if isinstance(outcome, Unmatched)
            )
            if question is not None
        )
        questions.extend(
            self.question_for_unfilled(role, instruction.speaker)
            for role in instruction.unfilled_roles
        )
        return questions

    def question_for_ambiguity(self, outcome: Ambiguity, person_id: str) -> Question:
        """A choice between the entities that matched, in the speaker's words."""
        choice = self.describer.choose_between(outcome.candidates)
        matched = [str(entity.name) for entity in outcome.candidates]
        if not choice.is_separable:
            return Question(
                text=f"I cannot tell the {choice.noun}s apart in words. "
                f"Which {choice.noun} do you mean?",
                answers_schema={str(outcome.role): {"type": "string"}},
                person_id=person_id,
                role=outcome.role,
                candidates=matched,
            )
        offered = list(choice.entity_per_option)
        return Question(
            text=f"Which {choice.noun} do you mean: {', or '.join(offered)}?",
            answers_schema={str(outcome.role): {"type": "string", "enum": offered}},
            person_id=person_id,
            role=outcome.role,
            options={
                option: str(entity.name)
                for option, entity in choice.entity_per_option.items()
            },
            candidates=matched,
        )

    def question_for_unmatched(
        self, outcome: Unmatched, person_id: str
    ) -> Optional[Question]:
        """Which place was meant, when the one described matched none: one of its
        type, or any place where the twin holds none of that type."""
        known = self.grounder.entities_of_type(outcome.constraint.entity_type)
        described = self.describer.describe_constraint(outcome.constraint)
        if known:
            text = f"I cannot find {described}. Which one do you mean?"
        else:
            known = self.grounder.searchable_entities()
            text = f"I do not know {described}. Which place do you mean?"
        return Question(
            text=text,
            answers_schema={str(outcome.role): {"type": "string"}},
            person_id=person_id,
            role=outcome.role,
            candidates=[str(entity.name) for entity in known],
            topic=f"which {outcome.role}",
        )

    def question_for_unfilled(self, role: ThematicRole, person_id: str) -> Question:
        """A request for a role the speaker left open: a place among the twin's, or
        the item in words."""
        return Question(
            text=str(RoleQuestion[role.name]),
            answers_schema={str(role): {"type": "string"}},
            person_id=person_id,
            candidates=(
                []
                if role == ThematicRole.OBJECT
                else [
                    str(entity.name) for entity in self.grounder.searchable_entities()
                ]
            ),
            role=role,
        )

    # %% feeding the answers back

    def resolve(
        self,
        instruction: ParsedInstruction,
        outcomes: List[GroundingOutcome],
        asked: Optional[Set[str]] = None,
    ) -> List[Answer]:
        """
        Ask every outstanding question not in `asked` and keep what came back.

        :param asked: The texts of the questions already put, which this adds to.
        """
        asked = set() if asked is None else asked
        answers = []
        for question in self.questions_for(instruction, outcomes):
            topic = question.topic or question.text
            if topic in asked:
                continue
            asked.add(topic)
            reply = self.skills.ask(question)
            if question.answer_key in reply:
                answers.append(
                    Answer(question=question, value=reply[question.answer_key])
                )
        return answers

    def apply(
        self, instruction: ParsedInstruction, answers: List[Answer]
    ) -> ParsedInstruction:
        """The instruction with the item an answer names in words, and with every
        place an answer names pinned to its role."""
        clarified = instruction
        for answer in answers:
            role = answer.question.role
            if role == ThematicRole.OBJECT:
                item = EntityConstraint(entity_type=None, spoken_form=str(answer.value))
                clarified = self.settled(clarified, role, item)
                continue
            entity = self.entity_for(answer)
            if entity is not None and str(entity.name) in answer.question.candidates:
                clarified = self.bind(clarified, role, entity)
        return clarified

    def choices_for(self, question: Question) -> Dict[str, SemanticAnnotation]:
        """What an answer to this question may mean, by the words for it."""
        if question.options:
            return {
                option: self.grounder.entity_named(name)
                for option, name in question.options.items()
            }
        return {
            self.describer.describe(entity).phrase: entity
            for entity in (
                self.grounder.entity_named(name) for name in question.candidates
            )
            if entity is not None
        }

    def entity_for(self, answer: Answer) -> Optional[SemanticAnnotation]:
        """The entity an answer names, among the ones its question offered."""
        choices = self.choices_for(answer.question)
        spoken = str(answer.value)
        by_name = {str(entity.name): entity for entity in choices.values()}
        matches = (
            choices.get(spoken),
            by_name.get(spoken),
            self.describer.entity_for_phrase(spoken, list(choices.values())),
        )
        literal = next((match for match in matches if match is not None), None)
        if literal is not None or self.interpreter is None:
            return literal
        return choices.get(
            self.interpreter.choose(answer.question.text, list(choices), spoken)
        )

    def bind(
        self,
        instruction: ParsedInstruction,
        role: ThematicRole,
        entity: SemanticAnnotation,
    ) -> ParsedInstruction:
        """The instruction with `role` narrowed to one named entity."""
        spoken = instruction.constraint_for(role)
        pinned = self.pinned_to(entity, spoken.spoken_form if spoken else None)
        return self.settled(instruction, role, pinned)

    @staticmethod
    def settled(
        instruction: ParsedInstruction,
        role: ThematicRole,
        constraint: EntityConstraint,
    ) -> ParsedInstruction:
        """The instruction with `role` filled by `constraint`, which the speaker
        settled."""
        return replace(
            instruction,
            roles=[binding for binding in instruction.roles if binding.role != role]
            + [RoleBinding(role=role, constraint=constraint)],
            unfilled_roles=[
                unfilled for unfilled in instruction.unfilled_roles if unfilled != role
            ],
            confidence_per_role=[
                scored
                for scored in instruction.confidence_per_role
                if scored.role != role
            ]
            + [RoleConfidence(role=role, confidence=1.0)],
        )

    @staticmethod
    def pinned_to(
        entity: SemanticAnnotation, spoken_form: Optional[str] = None
    ) -> EntityConstraint:
        """A constraint only `entity` matches, by its name."""
        return EntityConstraint(
            entity_type=type(entity).__name__,
            qualifiers=[
                Qualifier(attribute=QualifierAttribute.NAME, value=str(entity.name))
            ],
            spoken_form=spoken_form,
        )
