from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import EntityConstraint, Qualifier
from speech2plan.placement import PlacementResolver, QualifierAttribute

# %% words


def spoken_words(type_name: str) -> str:
    """`LivingRoom` as the words a person says."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", type_name).lower()


def plural(noun: str, count: int) -> str:
    """`noun` in the number `count` calls for."""
    return noun if count == 1 else f"{noun}s"


def spoken_list(phrases: List[str]) -> str:
    """`phrases` as a person would list them."""
    if len(phrases) < 2:
        return "".join(phrases)
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def definite(words: str) -> str:
    """`words` as a reference to a known entity."""
    return f"the {words}"


def indefinite(words: str) -> str:
    """`words` as a reference to one of several."""
    return f"{'an' if words[0] in 'aeiou' else 'a'} {words}"


def comparable(phrase: str) -> str:
    """A phrase stripped of what a person varies freely when saying it."""
    words = phrase.lower().replace(",", " ").split()
    return " ".join(word for word in words if word not in ("the", "a", "one"))


# %% one entity in words


@dataclass
class EntityDescription:
    """How a person would refer to one world entity."""

    entity: SemanticAnnotation = field(kw_only=True)
    noun: str = field(kw_only=True)
    """What kind of thing it is: 'kitchen', 'cup'."""

    feature: Optional[str] = field(kw_only=True, default=None)
    """What sets it apart from others of its kind: 'on the first floor'."""

    @property
    def phrase(self) -> str:
        """The full reference, as in 'the cup in the kitchen on the first floor'."""
        if self.feature is None:
            return f"the {self.noun}"
        return f"the {self.noun} {self.feature}"

    @property
    def option(self) -> Optional[str]:
        """This entity as one option in a question, or None when nothing sets it apart."""
        if self.feature is None:
            return None
        return f"the one {self.feature}"


# %% several of a kind


@dataclass
class Choice:
    """The options to offer for a reference that matched several entities."""

    noun: str = field(kw_only=True)
    """The kind of thing the candidates are, or 'one' when they differ."""

    entity_per_option: Dict[str, SemanticAnnotation] = field(kw_only=True)
    """The entity behind each spoken option, empty when words cannot separate them."""

    count: int = field(kw_only=True)

    @property
    def is_separable(self) -> bool:
        """True when every candidate has words of its own."""
        return bool(self.entity_per_option)


# %% the describer


@dataclass
class EntityDescriber:
    """Puts world entities into the words a person would use for them."""

    world: World = field(kw_only=True)
    resolver: Optional[PlacementResolver] = field(kw_only=True, default=None)

    def __post_init__(self):
        if self.resolver is None:
            self.resolver = PlacementResolver(world=self.world)

    def describe(self, entity: SemanticAnnotation) -> EntityDescription:
        """How to refer to one entity."""
        stated = self.stated_description(entity)
        if stated is not None:
            return EntityDescription(entity=entity, noun=stated)
        return EntityDescription(
            entity=entity,
            noun=spoken_words(type(entity).__name__),
            feature=self.feature_of(entity),
        )

    def stated_description(self, entity: SemanticAnnotation) -> Optional[str]:
        """The twin's own words for the entity, when it carries any."""
        body = self.resolver.body_of(entity)
        for annotation in self.world.get_semantic_annotations_by_type(
            NaturalLanguageDescription
        ):
            if annotation.root is body:
                return annotation.description
        return None

    def feature_of(self, entity: SemanticAnnotation) -> Optional[str]:
        """Where the entity is, in words, which is what tells namesakes apart."""
        if isinstance(entity, Room):
            storey = self.resolver.storey_of(entity)
            return None if storey is None else f"on the {spoken_words(storey)}"
        room = self.resolver.room_containing(entity)
        return None if room is None else f"in {self.describe(room).phrase}"

    def choose_between(self, entities: List[SemanticAnnotation]) -> Choice:
        """The options for a question offering these entities.

        Entities that share their words leave no options at all, because a
        choice nobody can answer is worse than an open question.
        """
        descriptions = [self.describe(entity) for entity in entities]
        nouns = {description.noun for description in descriptions}
        noun = nouns.pop() if len(nouns) == 1 else "one"
        options = {}
        for description in descriptions:
            spoken = description.option if noun != "one" else description.phrase
            if spoken is None or spoken in options:
                return Choice(noun=noun, entity_per_option={}, count=len(entities))
            options[spoken] = description.entity
        return Choice(noun=noun, entity_per_option=options, count=len(entities))

    def entity_for_phrase(
        self, phrase: str, entities: List[SemanticAnnotation]
    ) -> Optional[SemanticAnnotation]:
        """The entity whose own words match `phrase`, or None."""
        spoken = comparable(phrase)
        for entity in entities:
            description = self.describe(entity)
            said = {comparable(description.phrase)}
            if description.feature is not None:
                said.add(comparable(description.feature))
                said.add(comparable(description.option))
            if spoken in said:
                return entity
        return None

    def constraint_words(self, constraint: EntityConstraint) -> str:
        """A constraint in words, without an article."""
        words = [spoken_words(constraint.entity_type)]
        words.extend(
            phrase
            for phrase in (
                self.qualifier_phrase(qualifier) for qualifier in constraint.qualifiers
            )
            if phrase is not None
        )
        return " ".join(words)

    def describe_constraint(self, constraint: EntityConstraint) -> str:
        """A constraint as the speaker put it, as in 'the cup in the kitchen'."""
        return definite(self.constraint_words(constraint))

    def qualifier_phrase(self, qualifier: Qualifier) -> Optional[str]:
        """One qualifier in words, or None when it has none of its own."""
        if qualifier.attribute == QualifierAttribute.FLOOR:
            return f"on the {spoken_words(qualifier.value)}"
        if qualifier.attribute == QualifierAttribute.ROOM:
            room = self.resolver.room_named(qualifier.value)
            return None if room is None else f"in {self.describe(room).phrase}"
        return None
