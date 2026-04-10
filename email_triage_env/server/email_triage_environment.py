"""
Email Triage Environment — OpenEnv-compliant implementation.

Simulates an email support queue where an AI agent must classify,
prioritize, and draft replies for incoming emails.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation

from ..models import EmailAction, EmailObservation, EmailState
from .grader import grade_single_email
from .task_data import TASKS


class EmailTriageEnvironment(Environment):
    """
    Email Triage environment.

    Each episode loads a task (easy/medium/hard) and presents emails
    one at a time. The agent triages each email via step(). The episode
    ends when all emails are processed.

    Trajectory-aware scoring:
      - Consistency bonus: +0.05 if all emails scored above 0.5
      - Improvement bonus: +0.03 if scores trend upward across the episode
      - Short-reply penalty: -0.1 for replies under 3 words
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = EmailState()
        self._task = None
        self._scores: list[float] = []

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="Email Triage Environment",
            description=(
                "A real-world email support triage environment. "
                "Agents classify, prioritize, and draft replies for "
                "customer support emails across 3 difficulty levels."
            ),
            version="0.2.0",
            author="Hackathon Team",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id", "easy_triage")
        task_factory = TASKS.get(task_id)
        if task_factory is None:
            task_factory = TASKS["easy_triage"]
            task_id = "easy_triage"

        self._task = task_factory()
        self._scores = []
        self._state = EmailState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_email_index=0,
            total_emails=len(self._task.emails),
            cumulative_reward=0.0,
        )

        email = self._task.emails[0]
        return EmailObservation(
            task_id=self._task.task_id,
            email_from=email.sender,
            email_subject=email.subject,
            email_body=email.body,
            email_timestamp=email.timestamp,
            history=email.thread_history,
            feedback=f"Task: {self._task.description}",
            emails_remaining=len(self._task.emails),
            done=False,
            reward=None,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if not isinstance(action, EmailAction):
            raise ValueError(f"Expected EmailAction, got {type(action)}")

        if self._task is None:
            return EmailObservation(
                feedback="Environment not initialized. Call reset() first.",
                done=True,
                reward=0.001,
            )

        idx = self._state.current_email_index
        if idx >= len(self._task.emails):
            avg = (
                sum(self._scores) / len(self._scores) if self._scores else 0.0
            )
            return EmailObservation(
                feedback="All emails already processed.",
                done=True,
                reward=max(0.001, min(avg, 0.999)),
            )

        # Grade current email
        gold = self._task.emails[idx]
        reward = grade_single_email(action, gold)
        self._scores.append(reward)

        # Penalize clearly bad behaviour
        if len(action.reply.split()) < 3:
            reward = max(reward - 0.1, 0.001)

        self._state.step_count += 1
        self._state.current_email_index += 1
        self._state.cumulative_reward += reward

        # Check if episode is done
        done = self._state.current_email_index >= len(self._task.emails)

        if done:
            avg = sum(self._scores) / len(self._scores)

            # Trajectory bonuses
            bonus = 0.0
            # Consistency bonus: all emails scored above 0.5
            if all(s > 0.5 for s in self._scores):
                bonus += 0.05
            # Improvement bonus: scores trend upward
            if len(self._scores) >= 3:
                improving = sum(
                    1 for i in range(1, len(self._scores))
                    if self._scores[i] >= self._scores[i - 1]
                )
                if improving >= len(self._scores) - 1:
                    bonus += 0.03

            final_score = max(0.001, min(avg + bonus, 0.999))

            return EmailObservation(
                task_id=self._task.task_id,
                feedback=(
                    f"Episode complete. Average score: {avg:.4f}, "
                    f"trajectory bonus: +{bonus:.2f}, "
                    f"final: {final_score:.4f}. "
                    f"Individual scores: {self._scores}"
                ),
                emails_remaining=0,
                done=True,
                reward=final_score,
            )

        # Serve next email
        next_email = self._task.emails[self._state.current_email_index]
        remaining = len(self._task.emails) - self._state.current_email_index
        return EmailObservation(
            task_id=self._task.task_id,
            email_from=next_email.sender,
            email_subject=next_email.subject,
            email_body=next_email.body,
            email_timestamp=next_email.timestamp,
            history=next_email.thread_history,
            feedback=f"Previous email scored {reward:.4f}.",
            emails_remaining=remaining,
            done=False,
            reward=max(0.001, min(reward, 0.999)),
        )

    @property
    def state(self) -> EmailState:
        return self._state
