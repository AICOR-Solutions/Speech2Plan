from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Optional

from speech2plan.contract import (
    IntentName,
    ParsedInstruction,
    ThematicRole,
)
from speech2plan.description import bare_noun
from speech2plan.exceptions import LanguageModelCallFailed, UnusableReply
from speech2plan.grounding import Grounder, ResolvedRole
from speech2plan.intent_message import GroundingStatus
from speech2plan.parsing.backend import CompletionRequest, LanguageModel
from speech2plan.parsing.instruction_parser import InstructionParser
from speech2plan.settings import default_settings

# %% what is scored


class Stage(StrEnum):
    """One step from an utterance to grounded roles, scored on its own."""

    PARSED = "parsed"
    """A usable reply came back, after at most the one correction."""

    INTENT = "intent"
    ROLES = "roles"
    """Every place names the expected type, the item is the expected word, and no
    other role is filled."""

    UNFILLED = "unfilled roles"
    GROUNDING = "grounding"
    """Every expected role grounds as expected in the world, before any dialogue."""


@dataclass
class ExpectedGrounding:
    """How one role should ground, and to what if it resolves."""

    status: GroundingStatus = field(kw_only=True)
    entity: Optional[str] = field(kw_only=True, default=None)

    @classmethod
    def resolved(cls, entity: str) -> ExpectedGrounding:
        return cls(status=GroundingStatus.RESOLVED, entity=entity)

    @classmethod
    def ambiguous(cls) -> ExpectedGrounding:
        return cls(status=GroundingStatus.AMBIGUOUS)

    @classmethod
    def unmatched(cls) -> ExpectedGrounding:
        return cls(status=GroundingStatus.UNMATCHED)

    def __str__(self) -> str:
        return (
            str(self.status) if self.entity is None else f"{self.status} {self.entity}"
        )


@dataclass
class EvaluationCase:
    """One utterance and what it should become."""

    utterance: str = field(kw_only=True)
    intent: IntentName = field(kw_only=True)
    item: Optional[str] = field(kw_only=True, default=None)
    """The word for what the robot brings, without determiners, None when unsaid."""

    entity_type_per_role: Dict[ThematicRole, Optional[str]] = field(
        kw_only=True, default_factory=dict
    )
    """The type of each place role, None for a kind of place the world lacks."""

    unfilled_roles: List[ThematicRole] = field(kw_only=True, default_factory=list)
    grounding_per_role: Dict[ThematicRole, ExpectedGrounding] = field(
        kw_only=True, default_factory=dict
    )


@dataclass
class CaseResult:
    """How one case fared, stage by stage."""

    case: EvaluationCase = field(kw_only=True)
    passed: Dict[Stage, bool] = field(kw_only=True)
    failures: List[str] = field(kw_only=True, default_factory=list)
    """What went wrong, in words."""

    model_calls: int = field(kw_only=True, default=0)
    prompt_characters: int = field(kw_only=True, default=0)
    """System and user prompt of every call, summed."""

    @property
    def passed_all(self) -> bool:
        return all(self.passed.values())


# %% what a parse costs


@dataclass
class CountingModel(LanguageModel):
    """Passes requests on to a model, counting them and the prompt they carry."""

    model: LanguageModel = field(kw_only=True)
    calls: int = field(kw_only=True, default=0)
    prompt_characters: int = field(kw_only=True, default=0)

    def complete(self, request: CompletionRequest) -> str:
        self.calls += 1
        self.prompt_characters += len(request.system) + len(request.user)
        return self.model.complete(request)


# %% scoring


@dataclass
class Evaluation:
    """Parses each case and grounds the parse in one world, scoring every stage.

    The parser's model is wrapped to count what each case costs.
    """

    parser: InstructionParser = field(kw_only=True)
    grounder: Grounder = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.parser.model, CountingModel):
            self.parser.model = CountingModel(model=self.parser.model)

    def run(self, cases: List[EvaluationCase]) -> List[CaseResult]:
        return [self.score(case) for case in cases]

    def score(self, case: EvaluationCase) -> CaseResult:
        counter = self.parser.model
        calls, characters = counter.calls, counter.prompt_characters
        result = self.judge(case)
        result.model_calls = counter.calls - calls
        result.prompt_characters = counter.prompt_characters - characters
        return result

    def judge(self, case: EvaluationCase) -> CaseResult:
        try:
            instruction = self.parser.parse(case.utterance)
        except (UnusableReply, LanguageModelCallFailed) as failure:
            return CaseResult(
                case=case,
                passed={stage: False for stage in Stage},
                failures=[str(failure)],
            )
        types = {
            binding.role: binding.constraint.entity_type
            for binding in instruction.places
        }
        item = bare_noun(instruction.item, default_settings().words.determiners) or None
        filled = {binding.role for binding in instruction.roles}
        expected_filled = set(case.entity_type_per_role) | (
            {ThematicRole.OBJECT} if case.item is not None else set()
        )
        unfilled = set(instruction.unfilled_roles)
        grounding = self.grounding(instruction)
        wrong = {
            role: grounding.get(role)
            for role, expected in case.grounding_per_role.items()
            if grounding.get(role) != expected
        }
        failure_per_stage = {
            Stage.PARSED: None,
            Stage.INTENT: (
                None
                if instruction.intent == case.intent
                else f"intent {instruction.intent}"
            ),
            Stage.ROLES: (
                None
                if types == case.entity_type_per_role
                and item == case.item
                and filled == expected_filled
                else f"types {self.named(types)}, item {item!r}"
            ),
            Stage.UNFILLED: (
                None
                if unfilled == set(case.unfilled_roles)
                else f"unfilled {sorted(str(role) for role in unfilled)}"
            ),
            Stage.GROUNDING: f"grounding {self.named(wrong)}" if wrong else None,
        }
        return CaseResult(
            case=case,
            passed={
                stage: failure is None for stage, failure in failure_per_stage.items()
            },
            failures=[
                failure for failure in failure_per_stage.values() if failure is not None
            ],
        )

    def grounding(
        self, instruction: ParsedInstruction
    ) -> Dict[ThematicRole, ExpectedGrounding]:
        """How each filled role grounds, in the terms of an expectation."""
        return {
            outcome.role: ExpectedGrounding(
                status=GroundingStatus.of(outcome),
                entity=(
                    str(outcome.entity.name)
                    if isinstance(outcome, ResolvedRole)
                    else None
                ),
            )
            for outcome in self.grounder.ground(instruction)
        }

    @staticmethod
    def named(per_role: dict) -> str:
        """A per-role mapping for a failure message."""
        return (
            ", ".join(f"{role}={value}" for role, value in per_role.items()) or "none"
        )


def summary(results: List[CaseResult]) -> str:
    """One line per stage, then the end to end rate, as a table."""
    total = len(results)
    lines = [f"{'stage':<16}{'passed':>8}{'rate':>8}"]
    for stage in Stage:
        passed = sum(result.passed[stage] for result in results)
        lines.append(f"{stage:<16}{passed:>5}/{total:<2}{passed / total:>8.0%}")
    passed = sum(result.passed_all for result in results)
    lines.append(f"{'all stages':<16}{passed:>5}/{total:<2}{passed / total:>8.0%}")
    calls = sum(result.model_calls for result in results)
    characters = sum(result.prompt_characters for result in results)
    lines.append(f"model calls per case      {calls / total:.1f}")
    lines.append(f"prompt characters per call {characters / max(calls, 1):.0f}")
    return "\n".join(lines)


# %% the golden set, for the care ward


def evaluation_cases() -> List[EvaluationCase]:
    """What a nurse on the ground floor of the care ward might ask the transport
    robot: to go somewhere, or to bring something somewhere."""
    return [
        EvaluationCase(
            utterance="Go to the kitchen on the ground floor",
            intent=IntentName.MOVE_TO,
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("kitchen_ground")
            },
        ),
        EvaluationCase(
            utterance="Go to the lounge upstairs",
            intent=IntentName.MOVE_TO,
            entity_type_per_role={ThematicRole.GOAL: "LivingRoom"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("lounge")
            },
        ),
        EvaluationCase(
            utterance="Bring the insulin to the patient room",
            intent=IntentName.BRING_OBJECT,
            item="insulin",
            entity_type_per_role={ThematicRole.GOAL: "Bedroom"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("patient_room")
            },
        ),
        EvaluationCase(
            utterance="Bring the cup to the kitchen on the first floor",
            intent=IntentName.BRING_OBJECT,
            item="cup",
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("kitchen_first")
            },
        ),
        EvaluationCase(
            utterance="Take the gloves to the bathroom",
            intent=IntentName.BRING_OBJECT,
            item="gloves",
            entity_type_per_role={ThematicRole.GOAL: "Bathroom"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("bathroom")
            },
        ),
        EvaluationCase(
            utterance="Bring my pills to the living room",
            intent=IntentName.BRING_OBJECT,
            item="pills",
            entity_type_per_role={ThematicRole.GOAL: "LivingRoom"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("living_room")
            },
        ),
        EvaluationCase(
            utterance="Bring the blood samples upstairs to the kitchen",
            intent=IntentName.BRING_OBJECT,
            item="blood samples",
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("kitchen_first")
            },
        ),
        EvaluationCase(
            utterance="Put the water on the bedside table",
            intent=IntentName.BRING_OBJECT,
            item="water",
            entity_type_per_role={ThematicRole.GOAL: "SideTable"},
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("bedside_table")
            },
        ),
        EvaluationCase(
            utterance="Bring this to the kitchen",
            intent=IntentName.BRING_OBJECT,
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
            unfilled_roles=[ThematicRole.OBJECT],
            grounding_per_role={
                ThematicRole.GOAL: ExpectedGrounding.resolved("kitchen_ground")
            },
        ),
        EvaluationCase(
            utterance="Fetch the towels",
            intent=IntentName.BRING_OBJECT,
            item="towels",
            unfilled_roles=[ThematicRole.GOAL],
        ),
        EvaluationCase(
            utterance="Bring the bandages to the pharmacy",
            intent=IntentName.BRING_OBJECT,
            item="bandages",
            entity_type_per_role={ThematicRole.GOAL: None},
            grounding_per_role={ThematicRole.GOAL: ExpectedGrounding.unmatched()},
        ),
        EvaluationCase(
            utterance="Show the visitor to the living room",
            intent=IntentName.RAW_USER_INPUT,
        ),
    ]
