from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from speech2plan.contract import IntentName, ThematicRole

# %% expectation


@dataclass
class GoldenUtterance:
    """One utterance and the intent it should produce."""

    utterance: str = field(kw_only=True)
    intent: IntentName = field(kw_only=True)
    entity_type_per_role: Dict[ThematicRole, str] = field(
        kw_only=True, default_factory=dict
    )
    unfilled_roles: List[ThematicRole] = field(kw_only=True, default_factory=list)


# %% the set


def golden_utterances() -> List[GoldenUtterance]:
    return [
        GoldenUtterance(
            utterance="Bring this to the kitchen",
            intent=IntentName.BRING_OBJECT,
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
            unfilled_roles=[ThematicRole.OBJECT],
        ),
        GoldenUtterance(
            utterance="Bring the cup to the kitchen on the first floor",
            intent=IntentName.BRING_OBJECT,
            entity_type_per_role={
                ThematicRole.OBJECT: "Cup",
                ThematicRole.GOAL: "Kitchen",
            },
        ),
        GoldenUtterance(
            utterance="Pick up the plate",
            intent=IntentName.GRAB_OBJECT,
            entity_type_per_role={ThematicRole.OBJECT: "Plate"},
        ),
        GoldenUtterance(
            utterance="Put the cup on the table",
            intent=IntentName.PLACE_OBJECT,
            entity_type_per_role={
                ThematicRole.OBJECT: "Cup",
                ThematicRole.GOAL: "Table",
            },
        ),
        GoldenUtterance(
            utterance="Go to the kitchen on the ground floor",
            intent=IntentName.MOVE_TO,
            entity_type_per_role={ThematicRole.GOAL: "Kitchen"},
        ),
        GoldenUtterance(
            utterance="Show the visitor to the living room",
            intent=IntentName.GUIDE,
            entity_type_per_role={ThematicRole.GOAL: "LivingRoom"},
        ),
        GoldenUtterance(
            utterance="Put it down in the bathroom",
            intent=IntentName.PLACE_OBJECT,
            entity_type_per_role={ThematicRole.GOAL: "Bathroom"},
            unfilled_roles=[ThematicRole.OBJECT],
        ),
    ]
