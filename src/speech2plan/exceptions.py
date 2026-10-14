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


# %% errors


@dataclass
class LanguageModelCallFailed(Speech2PlanError):
    """A language model backend returned no usable reply."""

    command: str = field(kw_only=True)
    exit_code: int = field(kw_only=True)
    standard_error: str = field(kw_only=True)

    def error_message(self) -> str:
        return (
            f"{self.command} exited with {self.exit_code}. "
            f"Standard error: {self.standard_error.strip()}"
        )


@dataclass
class SymbolNotInVocabulary(Speech2PlanError):
    """The model named an entity type the world does not define."""

    entity_type: str = field(kw_only=True)
    known_entity_types: list[str] = field(kw_only=True)

    def error_message(self) -> str:
        return (
            f"'{self.entity_type}' is not a known entity type. "
            f"Known: {', '.join(sorted(self.known_entity_types))}"
        )


@dataclass
class ApiKeyNotFound(Speech2PlanError):
    """The environment file does not define the key."""

    key_name: str = field(kw_only=True)
    env_file: str = field(kw_only=True)

    def error_message(self) -> str:
        return f"{self.env_file} defines no {self.key_name}"
