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


# %% backend


@dataclass
class ClaudeCommandLineModel(LanguageModel):
    """Calls the locally installed Claude command line in print mode.

    Exists only for prototype reasons, so no api is needed
    """

    executable: str = field(kw_only=True, default="claude")
    timeout_seconds: float = field(kw_only=True, default=180.0)

    def complete(self, request: CompletionRequest) -> str:
        command = [
            self.executable,
            "--print",
            "--output-format",
            "json",
            "--append-system-prompt",
            request.system,
        ]
        completed = subprocess.run(
            command,
            input=self.render_user_message(request),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise LanguageModelCallFailed(
                command=" ".join(command[:4]),
                exit_code=completed.returncode,
                standard_error=completed.stderr,
            )
        envelope = json.loads(completed.stdout)
        if envelope[CommandLineOutputKey.IS_ERROR]:
            raise LanguageModelCallFailed(
                command=" ".join(command[:4]),
                exit_code=0,
                standard_error=envelope[CommandLineOutputKey.RESULT],
            )
        return envelope[CommandLineOutputKey.RESULT]

    def render_user_message(self, request: CompletionRequest) -> str:
        if not request.images:
            return request.user
        image_lines = "\n".join(f"Image: {path}" for path in request.images)
        return f"{image_lines}\n\n{request.user}"
