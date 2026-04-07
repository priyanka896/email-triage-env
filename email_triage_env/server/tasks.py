"""
Task definitions and email datasets for the Email Triage environment.

Three difficulty levels:
  - easy:   Clear-cut emails with obvious priority/category
  - medium: Ambiguous emails requiring nuance
  - hard:   Multi-threaded, conflicting signals, edge cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    timestamp: str
    thread_history: List[str] = field(default_factory=list)
    # Ground truth for grading
    gold_priority: str = "medium"
    gold_category: str = "general_inquiry"
    should_escalate: bool = False
    # Keywords that a good reply should address
    reply_must_mention: List[str] = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str  # easy, medium, hard
    emails: List[Email] = field(default_factory=list)
