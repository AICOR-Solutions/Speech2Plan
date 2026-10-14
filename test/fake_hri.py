from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# %% stand-ins for what pyhri tracks


@dataclass
class FakeVoice:
    """A tracked voice, which delivers speech the way `hri.Voice` does."""

    id: str = field(kw_only=True)
    locale: str = field(kw_only=True, default="en_US")
    speech_callbacks: List[Callable] = field(kw_only=True, default_factory=list)

    def on_speech(self, callback: Callable) -> None:
        self.speech_callbacks.append(callback)

    def say(self, utterance: str) -> None:
        """Deliver a final sentence to everyone listening."""
        for callback in self.speech_callbacks:
            callback(utterance)


@dataclass
class FakePerson:
    """A tracked person, with or without a voice bound to them."""

    id: str = field(kw_only=True)
    voice: Optional[FakeVoice] = field(kw_only=True, default=None)


@dataclass
class FakeListener:
    """Stands in for `hri.HRIListener`."""

    tracked_persons: Dict[str, FakePerson] = field(kw_only=True, default_factory=dict)
    voice_callbacks: List[Callable] = field(kw_only=True, default_factory=list)

    def on_voice(self, callback: Callable) -> None:
        self.voice_callbacks.append(callback)

    def spin_all(self, timeout) -> None:
        """Nothing to read: this listener is fed by `track`."""

    def track(self, voice: FakeVoice) -> FakeVoice:
        """Announce a newly tracked voice to everyone listening."""
        for callback in self.voice_callbacks:
            callback(voice)
        return voice
