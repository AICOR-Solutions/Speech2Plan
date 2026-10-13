from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from speech2plan.contract import EntityConstraint, Qualifier, ThematicRole
from speech2plan.grounding import Ambiguity, Grounder, ResolvedRole, Unmatched

# %% mimics


@dataclass
class MimicFloor:
    """Stands in for a semantic digital twin floor."""


@dataclass
class GroundFloor(MimicFloor): ...


@dataclass
class FirstFloor(MimicFloor): ...


@dataclass
class Kitchen:
    """Stands in for a kitchen annotation, with only what grounding reads."""

    name: str = field(kw_only=True)
    floor: MimicFloor = field(kw_only=True)


@dataclass
class MimicWorld:
    """A world that returns a fixed set of rooms."""

    rooms: List[Kitchen] = field(kw_only=True)

    def get_semantic_annotations_by_type(self, semantic_annotation_type) -> List:
        return self.rooms


# %% fixtures


@pytest.fixture
def two_kitchens() -> MimicWorld:
    return MimicWorld(
        rooms=[
            Kitchen(name="kitchen_ground", floor=GroundFloor()),
            Kitchen(name="kitchen_first", floor=FirstFloor()),
        ]
    )


# %% same-noun disambiguation


def test_two_kitchens_are_ambiguous(two_kitchens):
    grounder = Grounder(world=two_kitchens)
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL, EntityConstraint(entity_type="Kitchen")
    )
    assert isinstance(outcome, Ambiguity)
    assert outcome.role == ThematicRole.GOAL
    assert len(outcome.candidates) == 2


def test_floor_qualifier_resolves_the_ambiguity(two_kitchens):
    grounder = Grounder(world=two_kitchens)
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL,
        EntityConstraint(
            entity_type="Kitchen",
            qualifiers=[Qualifier(attribute="floor", value="GroundFloor")],
        ),
    )
    assert isinstance(outcome, ResolvedRole)
    assert outcome.entity.name == "kitchen_ground"


def test_unknown_room_type_does_not_match(two_kitchens):
    grounder = Grounder(world=two_kitchens)
    outcome = grounder.ground_constraint(
        ThematicRole.GOAL, EntityConstraint(entity_type="Bathroom")
    )
    assert isinstance(outcome, Unmatched)
