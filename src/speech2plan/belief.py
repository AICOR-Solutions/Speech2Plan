from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from speech2plan.contract import ParsedInstruction
from speech2plan.description import (
    EntityDescriber,
    plural,
    spoken_list,
    spoken_words,
)
from speech2plan.grounding import (
    Ambiguity,
    Grounder,
    GroundingOutcome,
    ResolvedRole,
    Unmatched,
)

# %% what the twin holds about an instruction


@dataclass
class BeliefReport:
    """States what the twin holds about the entities an instruction names."""

    grounder: Grounder = field(kw_only=True)
    describer: EntityDescriber = field(kw_only=True)
    most_entities: int = field(kw_only=True, default=3)
    """How many entities to name before falling back to counting them."""

    def sentences(
        self, instruction: ParsedInstruction, outcomes: List[GroundingOutcome]
    ) -> List[str]:
        """What bears on the roles the robot cannot act on yet.

        A resolved role is already named in the task itself, so it adds nothing.
        """
        return [
            sentence
            for sentence in (self.about(outcome) for outcome in outcomes)
            if sentence is not None
        ]

    def about(self, outcome: GroundingOutcome) -> Optional[str]:
        """One sentence about one role, or None when the role is settled."""
        if isinstance(outcome, Ambiguity):
            return self.about_candidates(outcome)
        if isinstance(outcome, Unmatched):
            return self.about_absence(outcome)
        if outcome.assumed:
            return self.about_assumption(outcome)
        return None

    def about_assumption(self, outcome: ResolvedRole) -> str:
        """What else the speaker could have meant, since the floor decided it."""
        others = [
            self.describer.describe(entity).phrase
            for entity in outcome.alternatives[: self.most_entities]
        ]
        verb = "is" if len(others) == 1 else "are"
        return f"There {verb} also {spoken_list(others)}."

    def about_candidates(self, outcome: Ambiguity) -> str:
        """Where the entities still in question are."""
        choice = self.describer.choose_between(outcome.candidates)
        known = len(self.grounder.entities_of_type(outcome.constraint.entity_type))
        counted = f"I know of {known} {plural(choice.noun, known)}"
        if known > choice.count:
            counted = f"{counted}, {choice.count} of them on this floor"
        if not choice.is_separable:
            shared = self.describer.describe(outcome.candidates[0]).feature
            return f"{counted}." if shared is None else f"{counted}, all {shared}."
        if choice.count > self.most_entities:
            return f"{counted}."
        offered = [
            option.removeprefix("the ") for option in choice.entity_per_option
        ]
        return f"{counted}: {spoken_list(offered)}."

    def about_absence(self, outcome: Unmatched) -> str:
        """What the twin holds of the kind that matched nothing."""
        known = self.grounder.entities_of_type(outcome.constraint.entity_type)
        noun = spoken_words(outcome.constraint.entity_type)
        if not known:
            return f"I know of no {noun} at all."
        counted = f"I know of {len(known)} {plural(noun, len(known))}"
        if len(known) > self.most_entities:
            return f"{counted}, none of them where you said."
        descriptions = [self.describer.describe(entity) for entity in known]
        where = [described.option or described.phrase for described in descriptions]
        return f"{counted}, {spoken_list(where)}."
