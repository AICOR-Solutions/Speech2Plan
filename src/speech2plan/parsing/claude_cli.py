from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum

from speech2plan.exceptions import LanguageModelCallFailed
from speech2plan.parsing.backend import CompletionRequest, LanguageModel

# %% envelope


class CommandLineOutputKey(StrEnum):
    """Keys of the JSON envelope the Claude command line prints."""

    RESULT = "result"
    IS_ERROR = "is_error"
    USAGE = "usage"


class InputTokenKey(StrEnum):
    """The usage entries that together are the prompt the model read."""

    UNCACHED = "input_tokens"
    CACHE_WRITTEN = "cache_creation_input_tokens"
    CACHE_READ = "cache_read_input_tokens"


# %% backend


@dataclass
class ClaudeCommandLineModel(LanguageModel):
    """Calls the locally installed Claude command line. Needs no API key.

    The request's system prompt replaces Claude Code's own, and no tools, MCP
    servers or settings are loaded: appended to the default instead, every call
    carried some 18k tokens of coding-agent prompt and tool definitions.
    """

    executable: str = field(kw_only=True, default="claude")
    timeout_seconds: float = field(kw_only=True, default=180.0)
    input_tokens: int = field(kw_only=True, default=0)
    """The prompt tokens of every call so far, cached ones included."""

    def command(self, request: CompletionRequest) -> list:
        return [
            self.executable,
            "--print",
            "--output-format",
            "json",
            "--system-prompt",
            request.system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--no-session-persistence",
        ]

    def complete(self, request: CompletionRequest) -> str:
        command = self.command(request)
        completed = subprocess.run(
            command,
            input=request.user,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise LanguageModelCallFailed(
                command=self.executable,
                exit_code=completed.returncode,
                standard_error=completed.stderr,
            )
        envelope = json.loads(completed.stdout)
        if envelope[CommandLineOutputKey.IS_ERROR]:
            raise LanguageModelCallFailed(
                command=self.executable,
                exit_code=0,
                standard_error=envelope[CommandLineOutputKey.RESULT],
            )
        usage = envelope.get(CommandLineOutputKey.USAGE) or {}
        self.input_tokens += sum(usage.get(key) or 0 for key in InputTokenKey)
        return envelope[CommandLineOutputKey.RESULT]
