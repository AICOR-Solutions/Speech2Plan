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


@dataclass
class SkillCallFailed(Speech2PlanError):
    """A ROS4HRI skill did not run."""

    skill: str = field(kw_only=True)
    reason: str = field(kw_only=True)

    def error_message(self) -> str:
        return f"{self.skill}: {self.reason}"


@dataclass
class IntentNotPlannable(Speech2PlanError):
    """No plan is built for this intent."""

    intent: str = field(kw_only=True)
    plannable: list[str] = field(kw_only=True)

    def error_message(self) -> str:
        return f"no plan for {self.intent}. Plannable: {', '.join(self.plannable)}"


@dataclass
class RoleNotGrounded(Speech2PlanError):
    """The intent names no world entity for a role the plan needs."""

    role: str = field(kw_only=True)
    intent: str = field(kw_only=True)

    def error_message(self) -> str:
        return f"{self.intent} needs an entity for '{self.role}', the intent names none"


@dataclass
class NoSurfaceInRoom(Speech2PlanError):
    """A room was given as where to put something, and it holds no surface."""

    room: str = field(kw_only=True)

    def error_message(self) -> str:
        return f"{self.room} has no supporting surface to put anything on"


@dataclass
class NoElevator(Speech2PlanError):
    """The plan changes storey and the world has no elevator."""

    level: str = field(kw_only=True)

    def error_message(self) -> str:
        return f"reaching {self.level} needs an elevator, the world has none"
