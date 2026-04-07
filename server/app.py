"""
FastAPI application for the Email Triage Environment.

Usage:
    uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

from ..models import EmailAction, EmailObservation
from .email_triage_environment import EmailTriageEnvironment

app = create_app(
    EmailTriageEnvironment,
    EmailAction,
    EmailObservation,
    env_name="email_triage_env",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
