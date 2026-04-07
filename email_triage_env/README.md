# Email Triage Environment

An OpenEnv-compliant environment that simulates a real-world email support triage workflow. An AI agent must classify, prioritize, and draft replies for incoming customer support emails.

## Motivation

Email triage is one of the most common knowledge-worker tasks. Support teams spend hours daily reading, categorizing, and responding to emails. This environment tests whether an AI agent can:

- Correctly identify email priority (urgent → low)
- Classify emails into the right category (bug, billing, spam, etc.)
- Draft helpful, professional replies
- Know when to escalate to a human

## Action Space

```python
class EmailAction(Action):
    priority: Priority    # "urgent" | "high" | "medium" | "low"
    category: Category    # "bug_report" | "feature_request" | "billing" |
                          # "account_access" | "general_inquiry" | "spam"
    reply: str            # Draft reply (1-2000 chars)
    escalate: bool        # Whether to escalate to human supervisor
```

## Observation Space

```python
class EmailObservation(Observation):
    task_id: str           # Current task identifier
    email_from: str        # Sender address
    email_subject: str     # Subject line
    email_body: str        # Email body
    email_timestamp: str   # ISO timestamp
    history: List[str]     # Previous thread messages
    feedback: str          # Grader feedback from last action
    emails_remaining: int  # Emails left in queue
    done: bool             # Episode finished?
    reward: float          # Score for last action (0.0-1.0)
```

## Tasks

| Task ID | Difficulty | Emails | Description |
|---------|-----------|--------|-------------|
| `easy_triage` | Easy | 5 | Clear-cut emails: login issue, feature request, billing question, app crash, data export inquiry |
| `medium_triage` | Medium | 5 | Ambiguous emails: performance bug, partnership inquiry, follow-up invoice, webhook failure, cancellation with chargeback threat |
| `hard_triage` | Hard | 5 | Complex emails: production data loss (legal threat), spam detection, mixed bug+feature thread, security vulnerability disclosure, angry escalation with thread history |

## Reward Function

Each email is scored on five dimensions (total 1.0):

| Component | Weight | Scoring |
|-----------|--------|---------|
| Priority | 0.25 | Exact match = 0.25, adjacent = 0.12, 2-away = 0.05 |
| Category | 0.25 | Exact match = 0.25, related category = 0.08 |
| Escalation | 0.10 | Correct = 0.10 |
| Reply Keywords | 0.20 | Proportion of required keywords mentioned |
| Reply Quality | 0.20 | Professionalism score (length, tone, sign-off) |

Reply quality sub-scoring:
- Length: 20+ words = 0.30, 10+ = 0.20, 5+ = 0.10
- Positive tone words (thank, apologize, investigate, etc.): up to 0.40
- Negative tone penalty (rude/unprofessional language): -0.20
- ALL CAPS penalty: -0.15
- Proper punctuation ending: +0.10
- Professional sign-off: +0.05

Trajectory bonuses (applied at episode end):
- Consistency bonus: +0.05 if all emails scored above 0.5
- Improvement bonus: +0.03 if scores trend upward across the episode

Penalties:
- Replies shorter than 3 words lose 0.1 points

Task score = average per-email score + trajectory bonuses (capped at 1.0).

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"

# Run the server
uvicorn email_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

### Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
export ENV_URL="http://localhost:8000"

python inference.py
```

## Baseline Scores

Scores will vary by model. Expected ranges:

| Task | Expected Score |
|------|---------------|
| easy_triage | 0.70 - 0.95 |
| medium_triage | 0.55 - 0.85 |
| hard_triage | 0.40 - 0.75 |

## API Endpoints

- `POST /reset` — Reset with `{"task_id": "easy_triage"}` (or medium/hard)
- `POST /step` — Submit action `{"action": {"priority": "...", "category": "...", "reply": "...", "escalate": false}}`
- `GET /state` — Current episode state
- `GET /health` — Health check
- `GET /schema` — Action/observation schemas
- `GET /metadata` — Environment metadata
- `WS /ws` — WebSocket for stateful sessions (used by inference script)
