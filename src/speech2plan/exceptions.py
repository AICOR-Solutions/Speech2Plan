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
class UnusableReply(Speech2PlanError):
    """The model's replies to an utterance broke the reply's rules, even when told how."""

    utterance: str = field(kw_only=True)
    problems: list[str] = field(kw_only=True)

    def error_message(self) -> str:
        return f"no usable parse of '{self.utterance}': {'; '.join(self.problems)}"


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
