from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from string import Template
from typing import Dict, FrozenSet, List

from speech2plan.contract import SpatialRelation

# %% the parts of the settings file


@dataclass(frozen=True)
class Prompts:
    """The text sent to the language models."""

    rules: str = field(kw_only=True)
    examples: str = field(kw_only=True)
    answer: str = field(kw_only=True)
    """How to read an answer, with `$question` and `$options` to fill in."""

    relations: Dict[SpatialRelation, str] = field(kw_only=True)
    """What each relation means."""

    def answer_to(self, question: str, offered: List[str]) -> str:
        options = "\n".join(f"- {option}" for option in offered)
        return Template(self.answer).substitute(question=question, options=options)


@dataclass(frozen=True)
class Words:
    """Word lists that speech is read with."""

    filler: FrozenSet[str] = field(kw_only=True)
    determiners: FrozenSet[str] = field(kw_only=True)
    optional_in_answers: FrozenSet[str] = field(kw_only=True)
    place_prepositions: FrozenSet[str] = field(kw_only=True)
    affirmative: FrozenSet[str] = field(kw_only=True)
    negative: FrozenSet[str] = field(kw_only=True)

    @staticmethod
    def of(sentence: str) -> List[str]:
        """The words of a sentence, lower case and without punctuation."""
        return re.findall(r"[a-z']+", sentence.lower())

    def is_filler(self, sentence: str) -> bool:
        """Whether the sentence holds no word but filler."""
        return set(self.of(sentence)) <= self.filler


# %% the file


@dataclass(frozen=True)
class Settings:
    prompts: Prompts = field(kw_only=True)
    words: Words = field(kw_only=True)

    @classmethod
    def load(cls, path: Path) -> Settings:
        with path.open("rb") as file:
            loaded = tomllib.load(file)
        prompts = dict(loaded["prompts"])
        prompts["relations"] = {
            SpatialRelation(name): meaning
            for name, meaning in prompts["relations"].items()
        }
        return cls(
            prompts=Prompts(**prompts),
            words=Words(
                **{name: frozenset(words) for name, words in loaded["words"].items()}
            ),
        )


@cache
def default_settings() -> Settings:
    """The settings file next to this module, read once."""
    return Settings.load(Path(__file__).with_name("settings.toml"))
