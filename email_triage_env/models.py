"""
Typed Pydantic models for the Email Triage environment.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM = "spam"


class EmailAction(Action):
    """Agent's triage decision for the current email."""

    priority: Priority = Field(..., description="Assigned priority level")
    category: Category = Field(..., description="Classified category")
    reply: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Draft reply to the sender",
    )
    escalate: bool = Field(
        default=False,
        description="Whether to escalate to a human supervisor",
    )


class EmailObservation(Observation):
    """What the agent sees: the current email and triage context."""

    task_id: str = Field(default="", description="Current task identifier")
    prompt: str = Field(default="", description="Instructions for the agent")
    email_from: str = Field(default="", description="Sender address")
    email_subject: str = Field(default="", description="Email subject line")
    email_body: str = Field(default="", description="Email body text")
    email_timestamp: str = Field(default="", description="When the email was sent")
    history: List[str] = Field(
        default_factory=list,
        description="Previous emails in the thread (if any)",
    )
    feedback: str = Field(
        default="",
        description="Feedback on the last action taken",
    )
    emails_remaining: int = Field(
        default=0,
        description="Number of emails left in the queue",
    )


class EmailState(State):
    """Tracks episode progress."""

    current_email_index: int = 0
    total_emails: int = 0
    cumulative_reward: float = 0.0
