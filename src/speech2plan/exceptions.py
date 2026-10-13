from __future__ import annotations

from dataclasses import dataclass, field

# %% base


@dataclass
class Speech2PlanError(Exception):
    """Base for every error raised by this package."""

    def error_message(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.error_message()


# %% language model errors


@dataclass
class LanguageModelCallFailed(Speech2PlanError):
    """The language model backend did not return a usable reply."""

    command: str = field(kw_only=True)
    exit_code: int = field(kw_only=True)
    standard_error: str = field(kw_only=True)

    def error_message(self) -> str:
        return (
            f"{self.command} exited with {self.exit_code}. "
            f"Standard error: {self.standard_error.strip()}"
        )


# %% grounding errors


@dataclass
class SymbolNotInVocabulary(Speech2PlanError):
    """The model named an entity type that the world does not define."""

    entity_type: str = field(kw_only=True)
    known_entity_types: list[str] = field(kw_only=True)

    def error_message(self) -> str:
        return (
            f"'{self.entity_type}' is not a known entity type. "
            f"Known: {', '.join(sorted(self.known_entity_types))}"
        )
