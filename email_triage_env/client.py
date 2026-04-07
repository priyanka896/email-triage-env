"""
Client wrapper for the Email Triage environment.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import EmailAction, EmailObservation, EmailState


class EmailTriageEnv(EnvClient[EmailAction, EmailObservation, EmailState]):

    def _step_payload(self, action: EmailAction) -> Dict[str, Any]:
        return {
            "priority": action.priority.value,
            "category": action.category.value,
            "reply": action.reply,
            "escalate": action.escalate,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[EmailObservation]:
        obs = EmailObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EmailState:
        return EmailState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_email_index=payload.get("current_email_index", 0),
            total_emails=payload.get("total_emails", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
