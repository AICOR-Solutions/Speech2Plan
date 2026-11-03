from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from speech2plan.contract import EntityConstraint, Qualifier, Relation
from speech2plan.placement import PlacementResolver, QualifierAttribute
from speech2plan.settings import Words, default_settings

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


def is_plural(noun: str) -> bool:
    """Whether a noun reads as plural, which is a guess from its last letters."""
    return noun.endswith("s") and not noun.endswith("ss")


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
    def words(self) -> str:
        """The reference without an article, as in 'cup in the kitchen'."""
        if self.feature is None:
            return self.noun
        return f"{self.noun} {self.feature}"

    @property
    def phrase(self) -> str:
        """The full reference, as in 'the cup in the kitchen on the first floor'."""
        return definite(self.words)

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
    words: Words = field(kw_only=True, default_factory=lambda: default_settings().words)
    """What a speaker says around the words for a thing."""

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

    def place_feature(self, entity: SemanticAnnotation) -> Optional[str]:
        """What the entity lies in or stands on, in words, as in 'on the table'."""
        container = self.resolver.container_of(entity)
        if container is not None:
            return f"in {definite(self.describe(container).noun)}"
        support = self.resolver.support_of(entity)
        if support is not None:
            return f"on {definite(self.describe(support).noun)}"
        return None

    def placed_room_feature(self, entity: SemanticAnnotation) -> Optional[str]:
        """What the entity is on or in, and in which room, as in
        'on the table in the kitchen on the ground floor'."""
        place, room = self.place_feature(entity), self.feature_of(entity)
        if place is None or room is None:
            return None
        return f"{place} {room}"

    def described_feature(self, entity: SemanticAnnotation) -> Optional[str]:
        """The feature the entity is described by, which the twin's own words leave out."""
        return self.describe(entity).feature

    def feature_levels(self) -> List[Callable[[SemanticAnnotation], Optional[str]]]:
        """The ways to set an entity apart, the shortest that works first."""
        return [self.described_feature, self.place_feature, self.placed_room_feature]

    def choose_between(self, entities: List[SemanticAnnotation]) -> Choice:
        """The options for a question offering these entities.

        Things of different kinds are told apart by what they are. Among things of
        one kind the room comes first, which is how a person tells most things
        apart; what they stand on or lie in comes next. Entities that share their words at every
        level leave no options at all, because a choice nobody can answer is worse
        than an open question.
        """
        descriptions = [self.describe(entity) for entity in entities]
        nouns = [description.noun for description in descriptions]
        if len(set(nouns)) == len(nouns) > 1:
            return Choice(
                noun="one",
                entity_per_option={
                    definite(noun): entity for noun, entity in zip(nouns, entities)
                },
                count=len(entities),
            )
        noun = nouns[0] if len(set(nouns)) == 1 else "one"
        for feature_for in self.feature_levels():
            features = [feature_for(entity) for entity in entities]
            if None in features:
                continue
            spoken = [
                (
                    f"the one {feature}"
                    if noun != "one"
                    else f"the {description.noun} {feature}"
                )
                for feature, description in zip(features, descriptions)
            ]
            if len(set(spoken)) == len(spoken):
                return Choice(
                    noun=noun,
                    entity_per_option=dict(zip(spoken, entities)),
                    count=len(entities),
                )
        return Choice(noun=noun, entity_per_option={}, count=len(entities))

    def entity_for_phrase(
        self, phrase: str, entities: List[SemanticAnnotation]
    ) -> Optional[SemanticAnnotation]:
        """The entity whose own words match `phrase`, or None."""
        spoken = self.comparable(phrase)
        words = spoken.split()
        heard = {spoken}
        if words and words[0] in self.words.place_prepositions:
            heard.add(" ".join(words[1:]))
        nouns = [self.describe(entity).noun for entity in entities]
        for entity in entities:
            description = self.describe(entity)
            said = {self.comparable(description.phrase)}
            if nouns.count(description.noun) == 1:
                said.add(self.comparable(description.noun))
            for feature_for in self.feature_levels():
                feature = feature_for(entity)
                if feature is not None:
                    said.add(self.comparable(feature))
                    said.add(self.comparable(f"the one {feature}"))
                    said.add(self.comparable(f"the {description.noun} {feature}"))
            if heard & said:
                return entity
        return None

    def entity_named(self, name: str) -> Optional[SemanticAnnotation]:
        """The annotation of that name, or None."""
        for annotation in self.world.semantic_annotations:
            if str(annotation.name) == name:
                return annotation
        return None

    def constraint_words(self, constraint: EntityConstraint) -> str:
        """A constraint in words, without an article.

        A constraint pinned to one entity by name is that entity's description.
        """
        for qualifier in constraint.qualifiers:
            if qualifier.attribute == QualifierAttribute.NAME:
                named = self.entity_named(qualifier.value)
                if named is not None:
                    return self.describe(named).words
        words = [self.noun_of(constraint)]
        words.extend(
            phrase
            for phrase in (
                self.qualifier_phrase(qualifier) for qualifier in constraint.qualifiers
            )
            if phrase is not None
        )
        words.extend(
            self.relation_phrase(relation) for relation in constraint.relations
        )
        return " ".join(words)

    def noun_of(self, constraint: EntityConstraint) -> str:
        """What the constraint describes, without where: 'cup', or the speaker's
        own word for a kind of thing the world lacks."""
        if constraint.entity_type is None:
            return self.spoken_noun(constraint.spoken_form)
        return spoken_words(constraint.entity_type)

    def spoken_noun(self, spoken_form: Optional[str]) -> str:
        """What the speaker called a thing, without the words before it: 'my pills'
        -> 'pills'."""
        words = (spoken_form or "").lower().split()
        while words and words[0] in self.words.determiners:
            words = words[1:]
        return " ".join(words) or "thing"

    def comparable(self, phrase: str) -> str:
        """A phrase stripped of what a person varies freely when saying it."""
        words = phrase.lower().replace(",", " ").split()
        return " ".join(
            word for word in words if word not in self.words.optional_in_answers
        )

    def relation_phrase(self, relation: Relation) -> str:
        """One relation in words, as in 'on the table in the kitchen'."""
        return (
            f"{relation.relation} {definite(self.constraint_words(relation.reference))}"
        )

    def describe_constraint(self, constraint: EntityConstraint) -> str:
        """A constraint as the speaker put it, as in 'the cup in the kitchen'."""
        return definite(self.constraint_words(constraint))

    def qualifier_phrase(self, qualifier: Qualifier) -> Optional[str]:
        """One qualifier in words, or None when it has none of its own."""
        if qualifier.attribute == QualifierAttribute.FLOOR:
            return f"on the {spoken_words(qualifier.value)}"
        return None
