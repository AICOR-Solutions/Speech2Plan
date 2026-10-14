from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from speech2plan.dialogue import DialogueSkills, Question

# %% a stand-in for the robot


@dataclass
class FakeDialogueSkills(DialogueSkills):
    """Records what was said and answers questions from a script."""

    scripted_answers: Dict[str, object] = field(kw_only=True, default_factory=dict)
    """Answers by key, so `goal` or `confirmed`."""

    said: List[str] = field(kw_only=True, default_factory=list)
    asked: List[Question] = field(kw_only=True, default_factory=list)

    def say(self, text: str, person_id: str) -> None:
        self.said.append(text)

    def ask(self, question: Question) -> Dict[str, object]:
        self.asked.append(question)
        key = question.answer_key
        if key not in self.scripted_answers:
            return {}
        return {key: self.scripted_answers[key]}
