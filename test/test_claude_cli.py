from __future__ import annotations

import json
import subprocess

from speech2plan.parsing.backend import CompletionRequest
from speech2plan.parsing.claude_cli import ClaudeCommandLineModel

# %% the command


def test_the_request_replaces_the_coding_agent_prompt_and_loads_no_tools():
    command = ClaudeCommandLineModel().command(
        CompletionRequest(system="You parse.", user="bring the cup")
    )
    assert command[command.index("--system-prompt") + 1] == "You parse."
    assert command[command.index("--tools") + 1] == ""
    assert "--append-system-prompt" not in command
    assert "--strict-mcp-config" in command


def test_the_prompt_tokens_of_every_call_are_counted(monkeypatch):
    envelope = {
        "result": "{}",
        "is_error": False,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 2000,
        },
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(envelope), stderr=""
        ),
    )
    model = ClaudeCommandLineModel()
    model.complete(CompletionRequest(system="s", user="u"))
    model.complete(CompletionRequest(system="s", user="u"))
    assert model.input_tokens == 4204
