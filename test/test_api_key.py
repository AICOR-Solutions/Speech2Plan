from __future__ import annotations

import pytest

from speech2plan.exceptions import ApiKeyNotFound
from speech2plan.parsing.openrouter import OpenRouterModel, read_api_key

# %% reading the key


def test_the_key_is_read_from_a_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=x\nOPENROUTER_API_KEY=sk-test-123\n")
    assert read_api_key(env_file, "OPENROUTER_API_KEY") == "sk-test-123"


def test_surrounding_quotes_are_dropped(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('OPENROUTER_API_KEY="sk-quoted"\n')
    assert read_api_key(env_file, "OPENROUTER_API_KEY") == "sk-quoted"


def test_a_missing_key_names_the_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SOMETHING_ELSE=1\n")
    with pytest.raises(ApiKeyNotFound) as raised:
        read_api_key(env_file, "OPENROUTER_API_KEY")
    assert "OPENROUTER_API_KEY" in str(raised.value)
    assert str(env_file) in str(raised.value)


# %% the backend uses it


def test_the_backend_defaults_to_the_repository_env_file():
    assert OpenRouterModel().env_file.name == ".env"
