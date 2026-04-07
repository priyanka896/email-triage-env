# Email Triage Environment — Complete Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Data Models (Pydantic Types)](#4-data-models)
5. [Environment Core Logic](#5-environment-core-logic)
6. [Grading System](#6-grading-system)
7. [Task & Email Dataset Design](#7-task--email-dataset-design)
8. [Server Layer (FastAPI + WebSocket)](#8-server-layer)
9. [Client Library](#9-client-library)
10. [Inference Script](#10-inference-script)
11. [Containerization (Docker)](#11-containerization)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment Guide](#13-deployment-guide)
14. [Design Decisions & Tradeoffs](#14-design-decisions--tradeoffs)

---

## 1. Project Overview

This project implements a real-world **Email Triage** environment using the
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates a
customer support email queue where an AI agent must:

- Read incoming emails one at a time
- Classify each email by priority and category
- Draft a professional reply
- Decide whether to escalate to a human supervisor

The environment follows the Gymnasium-style API (`step()` / `reset()` / `state()`)
and is fully compliant with the OpenEnv specification.

### Why Email Triage?

Email triage is a genuine, high-volume knowledge-worker task. Unlike games or
toy problems, it requires:

- **Reading comprehension** — understanding sender intent from unstructured text
- **Multi-dimensional classification** — priority AND category simultaneously
- **Generation** — drafting contextually appropriate replies
- **Judgment** — knowing when to escalate vs. handle independently
- **Thread awareness** — using conversation history for context

This makes it an ideal benchmark for evaluating LLM agent capabilities in a
realistic setting.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE LAYER                          │
│                                                                 │
│  inference.py                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐      │
│  │  OpenAI   │───>│  Parse   │───>│  Sanitize & Send     │      │
│  │  Client   │    │  JSON    │    │  via WebSocket       │      │
│  └──────────┘    └──────────┘    └──────────┬───────────┘      │
│                                              │                  │
└──────────────────────────────────────────────┼──────────────────┘
                                               │ WebSocket /ws
                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVER LAYER                             │
│                                                                 │
│  FastAPI Application (server/app.py)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  create_app(EmailTriageEnvironment, EmailAction,         │   │
│  │             EmailObservation, env_name="email_triage")   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Endpoints:                                                     │
│    POST /reset    — Start new episode (stateless HTTP)          │
│    POST /step     — Execute action (stateless HTTP)             │
│    GET  /state    — Current state                               │
│    GET  /health   — Health check                                │
│    GET  /schema   — JSON schemas for action/observation/state   │
│    GET  /metadata — Environment metadata                        │
│    WS   /ws       — Stateful WebSocket sessions                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT LAYER                           │
│                                                                 │
│  EmailTriageEnvironment (Environment base class)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  reset(task_id) ──> Load task, serve first email         │   │
│  │  step(action)   ──> Grade action, serve next email       │   │
│  │  state          ──> Return EmailState                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GRADING ENGINE (grader.py)                              │   │
│  │                                                          │   │
│  │  grade_single_email(action, gold_email) -> 0.0-1.0       │   │
│  │    ├── Priority match     (0.25)                         │   │
│  │    ├── Category match     (0.25)                         │   │
│  │    ├── Escalation match   (0.10)                         │   │
│  │    ├── Reply keywords     (0.20)                         │   │
│  │    └── Reply quality      (0.20)                         │   │
│  │                                                          │   │
│  │  Trajectory bonuses (at episode end):                    │   │
│  │    ├── Consistency bonus  (+0.05)                        │   │
│  │    └── Improvement bonus  (+0.03)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TASK DATA (task_data.py)                                │   │
│  │                                                          │   │
│  │  easy_triage   ──> 5 clear-cut emails                    │   │
│  │  medium_triage ──> 5 ambiguous emails                    │   │
│  │  hard_triage   ──> 5 complex emails with edge cases      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA MODEL LAYER                          │
│                                                                 │
│  models.py (Pydantic BaseModel subclasses)                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐     │
│  │  EmailAction  │  │ EmailObservation │  │  EmailState  │     │
│  │  (Action)     │  │ (Observation)    │  │  (State)     │     │
│  │               │  │                  │  │              │     │
│  │  priority     │  │  task_id         │  │  episode_id  │     │
│  │  category     │  │  email_from      │  │  step_count  │     │
│  │  reply        │  │  email_subject   │  │  email_index │     │
│  │  escalate     │  │  email_body      │  │  total       │     │
│  │               │  │  email_timestamp │  │  cumulative  │     │
│  │               │  │  history         │  │              │     │
│  │               │  │  feedback        │  │              │     │
│  │               │  │  emails_remaining│  │              │     │
│  │               │  │  done, reward    │  │              │     │
│  └──────────────┘  └──────────────────┘  └──────────────┘     │
│                                                                 │
│  Enums: Priority (urgent|high|medium|low)                       │
│         Category (bug_report|feature_request|billing|           │
│                   account_access|general_inquiry|spam)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Step)

```
1. Agent receives EmailObservation (email text + context)
2. Agent produces EmailAction (priority, category, reply, escalate)
3. Environment passes action to grader
4. Grader compares action against gold-standard labels
5. Grader scores across 5 dimensions → reward (0.0-1.0)
6. Environment advances to next email or ends episode
7. Agent receives new EmailObservation with reward feedback
```

### Session Management

```
HTTP Mode (stateless):
  Each POST /reset and POST /step creates a NEW environment instance.
  No state is preserved between requests.
  Useful for: health checks, schema inspection, simple testing.

WebSocket Mode (stateful):
  A single WS /ws connection maintains ONE environment instance.
  State persists across reset → step → step → ... → done.
  Used by: inference script, RL training loops, interactive sessions.
```


---

## 3. Project Structure

```
project-root/
│
├── Dockerfile                          # Root Dockerfile for HF Spaces
├── inference.py                        # Baseline inference script (OpenAI client)
├── requirements.txt                    # Python dependencies
├── test_all.py                         # 14 comprehensive unit tests
├── DOCUMENTATION.md                    # This file
├── CHAT_HISTORY.txt                    # Development history
├── DEPLOYMENT_AND_TESTING_GUIDE.txt    # Step-by-step deployment guide
│
├── email_triage_env/                   # Main environment package
│   ├── __init__.py                     # Package exports
│   ├── models.py                       # Pydantic data models (Action/Observation/State)
│   ├── client.py                       # EnvClient subclass for WebSocket interaction
│   ├── openenv.yaml                    # OpenEnv manifest file
│   ├── pyproject.toml                  # Package configuration & dependencies
│   ├── README.md                       # Environment documentation
│   │
│   └── server/                         # Server-side code
│       ├── __init__.py                 # Server package marker
│       ├── app.py                      # FastAPI application entry point
│       ├── email_triage_environment.py # Core Environment implementation
│       ├── grader.py                   # Deterministic scoring engine
│       ├── tasks.py                    # Email & Task dataclass definitions
│       ├── task_data.py                # 15 emails across 3 difficulty levels
│       └── Dockerfile                  # Alternative Dockerfile (inside server/)
│
└── openEnv/                            # Cloned OpenEnv framework (reference only)
    └── ...                             # Not part of submission
```

### File Responsibilities

| File | Lines | Purpose |
|------|-------|---------|
| `models.py` | ~65 | Type-safe Pydantic models: EmailAction, EmailObservation, EmailState, Priority/Category enums |
| `email_triage_environment.py` | ~130 | Core environment: reset(), step(), state, get_metadata(), trajectory bonuses |
| `grader.py` | ~120 | 5-dimension scoring engine + reply quality analyzer |
| `task_data.py` | ~250 | 15 emails with ground truth labels across 3 tasks |
| `tasks.py` | ~30 | Email and Task dataclass definitions |
| `app.py` | ~25 | FastAPI app creation via create_app() |
| `client.py` | ~45 | WebSocket client for programmatic access |
| `inference.py` | ~180 | LLM-powered agent using OpenAI client |

---

## 4. Data Models

### EmailAction (extends openenv Action)

```python
class EmailAction(Action):
    priority: Priority      # Enum: urgent | high | medium | low
    category: Category      # Enum: bug_report | feature_request | billing |
                            #       account_access | general_inquiry | spam
    reply: str              # 1-2000 characters, the draft reply
    escalate: bool          # Whether to escalate to human supervisor
```

The Action base class from OpenEnv provides:
- `model_config = ConfigDict(extra="forbid")` — rejects unknown fields
- `metadata: Dict[str, Any]` — optional metadata dict

### EmailObservation (extends openenv Observation)

```python
class EmailObservation(Observation):
    task_id: str            # "easy_triage" | "medium_triage" | "hard_triage"
    email_from: str         # Sender email address
    email_subject: str      # Subject line
    email_body: str         # Full email body text
    email_timestamp: str    # ISO 8601 timestamp
    history: List[str]      # Previous messages in thread (for follow-ups)
    feedback: str           # Grader feedback from previous action
    emails_remaining: int   # How many emails left in the queue
```

The Observation base class provides:
- `done: bool` — whether the episode has ended
- `reward: float | None` — reward signal from last action
- `metadata: Dict[str, Any]` — optional metadata

### EmailState (extends openenv State)

```python
class EmailState(State):
    current_email_index: int = 0    # Which email we're on (0-indexed)
    total_emails: int = 0           # Total emails in this task
    cumulative_reward: float = 0.0  # Sum of all rewards so far
```

The State base class provides:
- `episode_id: Optional[str]` — unique episode identifier
- `step_count: int` — number of steps taken

### Enums

```python
class Priority(str, Enum):
    URGENT = "urgent"    # Production outages, data loss, security
    HIGH = "high"        # Blocking issues, billing disputes
    MEDIUM = "medium"    # General bugs, partnership inquiries
    LOW = "low"          # Feature requests, spam, nice-to-haves

class Category(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    GENERAL_INQUIRY = "general_inquiry"
    SPAM = "spam"
```

---

## 5. Environment Core Logic

### Class: EmailTriageEnvironment

Extends `openenv.core.env_server.interfaces.Environment`.

```
SUPPORTS_CONCURRENT_SESSIONS = True
```

This flag tells the OpenEnv HTTP server that multiple WebSocket connections
can safely run their own environment instances simultaneously.

### reset(seed, episode_id, **kwargs)

```
Input:  kwargs["task_id"] = "easy_triage" | "medium_triage" | "hard_triage"
Output: EmailObservation with the first email in the queue

Steps:
  1. Look up task factory from TASKS dict
  2. If invalid task_id, fall back to "easy_triage"
  3. Call factory to get Task object with list of Email objects
  4. Initialize fresh EmailState (step_count=0, index=0)
  5. Clear score history
  6. Return observation with first email's fields populated
```

### step(action)

```
Input:  EmailAction with priority, category, reply, escalate
Output: EmailObservation with next email OR episode summary

Steps:
  1. Validate action is EmailAction (raise ValueError otherwise)
  2. Check environment is initialized (return error obs if not)
  3. Get current email from task by index
  4. Call grade_single_email(action, gold_email) → reward (0.0-1.0)
  5. Apply short-reply penalty (-0.1 if reply < 3 words)
  6. Append reward to scores list
  7. Advance state (step_count++, email_index++, cumulative_reward += reward)
  8. If all emails processed:
     a. Calculate average score
     b. Apply trajectory bonuses:
        - Consistency: +0.05 if ALL scores > 0.5
        - Improvement: +0.03 if scores trend upward
     c. Return done=True observation with final score
  9. Otherwise: return observation with next email
```

### state (property)

```
Returns: EmailState with episode_id, step_count, current_email_index,
         total_emails, cumulative_reward
```

### get_metadata()

```
Returns: EnvironmentMetadata(
    name="Email Triage Environment",
    description="A real-world email support triage environment...",
    version="0.2.0",
    author="Hackathon Team",
)
```

---

## 6. Grading System

### Architecture

The grading system is fully deterministic — no LLM calls, no randomness.
This ensures reproducible scores across runs.

### grade_single_email(action, gold) → float

Five scoring dimensions, each with partial credit:

#### 1. Priority Match (0.25 max)

```
Ordered scale: low(0) → medium(1) → high(2) → urgent(3)

Exact match:     0.25
Adjacent (±1):   0.12
Two-away (±2):   0.05
Three-away (±3): 0.00
```

Example: Gold is "urgent", agent says "high" → 0.12 (adjacent)

#### 2. Category Match (0.25 max)

```
Exact match:     0.25
Related pair:    0.08
Unrelated:       0.00

Related pairs:
  bug_report ↔ feature_request  (both are product feedback)
  billing ↔ account_access      (both are account-related)
```

Example: Gold is "bug_report", agent says "feature_request" → 0.08

#### 3. Escalation Match (0.10 max)

```
Correct:  0.10
Wrong:    0.00
```

Binary — either the agent correctly identifies escalation need or not.

#### 4. Reply Keywords (0.20 max)

```
Each email has a list of required keywords (e.g., ["password", "account", "reset"])
Score = 0.20 × (keywords_found / total_keywords)
```

Case-insensitive substring matching. Partial matches count
(e.g., "escalat" matches "escalated", "escalating").

#### 5. Reply Quality (0.20 max)

```
Score = 0.20 × _score_reply_quality(reply)

_score_reply_quality returns 0.0-1.0 based on:

Length (0.0-0.30):
  20+ words: 0.30
  10-19 words: 0.20
  5-9 words: 0.10
  <5 words: 0.00

Positive tone (0.0-0.40):
  Checks for: "thank", "apolog", "understand", "help", "assist",
  "looking into", "investigate", "resolve", "follow up", "please",
  "happy to", "glad to", "let me", "right away", "priority", "team"
  Score = 0.40 × min(hits / 4, 1.0)

Negative tone penalty (-0.20):
  Checks for: "stupid", "dumb", "idiot", "ridiculous", "whatever",
  "not my problem", "deal with it", "figure it out", "lol", "lmao"

ALL CAPS penalty (-0.15):
  If entire reply is uppercase (shouting)

Proper ending (+0.10):
  Reply ends with . ! or ?

Sign-off bonus (+0.05):
  Contains "regards", "best", "thank", "sincerely", "cheers"

Repetition penalty (-0.15):
  If >50% of sentences are duplicates
```

### Trajectory Bonuses (Episode-Level)

Applied only when the episode ends (all emails processed):

```
Consistency bonus (+0.05):
  Awarded if EVERY email in the episode scored above 0.5.
  Rewards agents that perform reliably across all emails.

Improvement bonus (+0.03):
  Awarded if scores trend upward (each score >= previous).
  Rewards agents that learn from feedback during the episode.

Final score = min(average_score + bonuses, 1.0)
```

### Penalties

```
Short reply penalty (-0.10):
  Applied if reply has fewer than 3 words.
  Prevents trivial "ok" or "thanks" responses.
```


---

## 7. Task & Email Dataset Design

### Design Principles

1. **Realistic content** — Emails read like actual support tickets
2. **Clear ground truth** — Each email has unambiguous correct labels
3. **Progressive difficulty** — Easy → Medium → Hard
4. **Diverse categories** — All 6 categories represented across tasks
5. **Thread context** — Some emails include conversation history
6. **Edge cases** — Spam, legal threats, mixed bug+feature, security disclosures

### Task: easy_triage (5 emails)

| # | Subject | Priority | Category | Escalate | Key Challenge |
|---|---------|----------|----------|----------|---------------|
| 1 | Cannot log in to my account | urgent | account_access | No | Clear urgency signal ("deadline today") |
| 2 | Add dark mode please | low | feature_request | No | Obvious feature request |
| 3 | Billing charge I don't recognize | high | billing | No | Straightforward billing question |
| 4 | App crashes on startup | high | bug_report | No | Clear bug with version info |
| 5 | How do I export my data? | low | general_inquiry | No | Simple how-to question |

### Task: medium_triage (5 emails)

| # | Subject | Priority | Category | Escalate | Key Challenge |
|---|---------|----------|----------|----------|---------------|
| 1 | Slow performance after update | high | bug_report | No | Could be bug or infra issue |
| 2 | Partnership opportunity | medium | general_inquiry | Yes | Escalation for business opportunity |
| 3 | Re: Invoice #8832 | high | billing | No | Thread history, follow-up context |
| 4 | Webhook notifications stopped | high | bug_report | No | Technical, could be maintenance-related |
| 5 | Cancellation request — unhappy | high | billing | Yes | Chargeback threat = escalation |

### Task: hard_triage (5 emails)

| # | Subject | Priority | Category | Escalate | Key Challenge |
|---|---------|----------|----------|----------|---------------|
| 1 | URGENT: Data loss after migration | urgent | bug_report | Yes | Legal threat + VIP + data loss |
| 2 | Congrats! You've won a $500 gift card | low | spam | No | Spam detection |
| 3 | Re: Re: Re: Feature + Bug combo | high | bug_report | No | Mixed signals: bug AND feature request |
| 4 | Security vulnerability in your API | urgent | bug_report | Yes | Responsible disclosure, time pressure |
| 5 | Re: Account upgrade — still waiting | urgent | billing | Yes | Angry repeat customer, thread history |

### Email Dataclass

```python
@dataclass
class Email:
    sender: str                          # Sender address
    subject: str                         # Subject line
    body: str                            # Full email body
    timestamp: str                       # ISO 8601
    thread_history: List[str] = []       # Previous messages
    gold_priority: str = "medium"        # Ground truth priority
    gold_category: str = "general_inquiry"  # Ground truth category
    should_escalate: bool = False        # Ground truth escalation
    reply_must_mention: List[str] = []   # Required keywords in reply
```

---

## 8. Server Layer

### app.py

```python
from openenv.core.env_server.http_server import create_app

app = create_app(
    EmailTriageEnvironment,   # Factory (class, not instance)
    EmailAction,              # Action type for deserialization
    EmailObservation,         # Observation type for serialization
    env_name="email_triage_env",
)
```

`create_app()` is provided by the OpenEnv framework. It:
1. Creates a FastAPI application
2. Wraps the environment in an HTTPEnvServer
3. Registers all HTTP and WebSocket endpoints
4. Handles serialization/deserialization automatically

### Endpoint Details

| Endpoint | Method | Stateful? | Description |
|----------|--------|-----------|-------------|
| `/health` | GET | No | Returns `{"status": "healthy"}` |
| `/metadata` | GET | No | Returns environment name, description, version |
| `/schema` | GET | No | Returns JSON schemas for action/observation/state |
| `/reset` | POST | No | Creates new env, calls reset(), destroys env |
| `/step` | POST | No | Creates new env, calls step(), destroys env |
| `/state` | GET | No | Creates new env, returns state, destroys env |
| `/ws` | WS | Yes | Persistent session with dedicated env instance |

### WebSocket Protocol

Messages are JSON with a `type` field:

```json
// Client → Server: Reset
{"type": "reset", "data": {"task_id": "easy_triage"}}

// Server → Client: Observation
{"type": "observation", "data": {"observation": {...}, "reward": 0.0, "done": false}}

// Client → Server: Step
{"type": "step", "data": {"priority": "urgent", "category": "bug_report", "reply": "...", "escalate": true}}

// Client → Server: State
{"type": "state"}

// Client → Server: Close
{"type": "close"}
```

### Serialization

The OpenEnv framework's `serialize_observation()` function:
- Calls `observation.model_dump(exclude={"reward", "done", "metadata"})`
- Places `reward` and `done` at the envelope level
- Result: `{"observation": {...}, "reward": float, "done": bool}`

---

## 9. Client Library

### EmailTriageEnv (extends EnvClient)

```python
class EmailTriageEnv(EnvClient[EmailAction, EmailObservation, EmailState]):
    def _step_payload(self, action: EmailAction) -> dict:
        # Converts EmailAction to JSON dict for WebSocket
        return {
            "priority": action.priority.value,
            "category": action.category.value,
            "reply": action.reply,
            "escalate": action.escalate,
        }

    def _parse_result(self, payload: dict) -> StepResult[EmailObservation]:
        # Parses server response into StepResult
        obs = EmailObservation(**payload["observation"])
        return StepResult(observation=obs, reward=payload.get("reward"), done=...)

    def _parse_state(self, payload: dict) -> EmailState:
        # Parses state response into EmailState
        return EmailState(episode_id=..., step_count=..., ...)
```

### Usage

```python
# Async (recommended)
async with EmailTriageEnv(base_url="http://localhost:8000") as client:
    result = await client.reset(task_id="easy_triage")
    result = await client.step(EmailAction(...))

# Sync wrapper
with EmailTriageEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset(task_id="easy_triage")
    result = client.step(EmailAction(...))
```

---

## 10. Inference Script

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   System     │     │   LLM API    │     │  Environment │
│   Prompt     │────>│  (OpenAI)    │────>│  (WebSocket) │
│              │     │              │     │              │
│  Guidelines  │     │  JSON output │     │  Grade &     │
│  for triage  │     │  parsed      │     │  next email  │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Flow per Task

```
1. Connect to environment via WebSocket
2. Send reset with task_id
3. Receive first email observation
4. Loop until done:
   a. Build user message from observation (email text + feedback + history)
   b. Call LLM with system prompt + conversation history
   c. Parse JSON from LLM response (handles markdown fences)
   d. Sanitize action fields (validate enums, truncate reply)
   e. Send action via WebSocket
   f. Receive observation with reward and next email
5. Print final score
```

### Configuration (Environment Variables)

```
API_BASE_URL   = "https://router.huggingface.co/v1"  # LLM endpoint
MODEL_NAME     = "Qwen/Qwen2.5-72B-Instruct"        # Model to use
HF_TOKEN       = "hf_..."                             # API key
ENV_URL        = "http://localhost:8000"               # Environment server
```

### System Prompt Design

The system prompt instructs the LLM to:
1. Output ONLY a JSON object (no explanation text)
2. Use exact enum values for priority and category
3. Follow specific guidelines for each priority level
4. Know when to escalate (legal threats, VIP, data loss, partnerships)
5. Draft professional, helpful replies addressing the sender's concern

### Error Handling

- LLM errors: falls back to default "medium/general_inquiry" action
- JSON parse errors: falls back to empty dict, sanitizer fills defaults
- Invalid enum values: sanitizer maps to closest valid value
- Empty replies: sanitizer provides generic professional reply

---

## 11. Containerization

### Dockerfile

```dockerfile
FROM python:3.11-slim

# System deps + uv package manager
RUN apt-get update && apt-get install -y build-essential curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY email_triage_env/pyproject.toml ./
COPY email_triage_env/ ./email_triage_env/
COPY email_triage_env/openenv.yaml ./

# Install with uv (faster than pip)
RUN uv venv /app/.venv && uv pip install -e "."

# Run FastAPI server
CMD ["uvicorn", "email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Resource Requirements

- CPU: 2 vCPU sufficient (no GPU needed)
- Memory: <1GB (pure Python, no ML models loaded)
- Disk: ~200MB (Python + dependencies)
- Startup time: ~5 seconds

---

## 12. Testing Strategy

### Test Suite: test_all.py (14 tests)

| Test | What It Verifies |
|------|-----------------|
| Easy Task | All 5 emails with correct answers score >0.85 |
| Medium Task | All 5 emails with correct answers score >0.70 |
| Hard Task | All 5 emails with correct answers score >0.70 |
| Partial Scoring | Wrong answers score low, adjacent priorities get partial credit |
| Reply Quality | Professional replies score high, rude/short replies score low |
| Related Category | Related categories (bug↔feature) get partial credit |
| Trajectory Bonus | Consistency bonus appears in feedback when all scores >0.5 |
| State Tracking | step_count, email_index, cumulative_reward update correctly |
| Reset Clears State | New reset zeroes all state, loads new task |
| Invalid Task Fallback | Unknown task_id falls back to easy_triage |
| Step Without Reset | Returns error observation gracefully |
| Score Range | All scores across all tasks are in 0.0-1.0 |
| Metadata | get_metadata() returns correct name, version, description |
| Grader Components | Individual grading dimensions score correctly in isolation |

### Running Tests

```bash
python test_all.py
# Expected: 14 passed, 0 failed
```

---

## 13. Deployment Guide

### Quick Deploy to Hugging Face Spaces

```bash
# 1. Login
huggingface-cli login

# 2. Create Space
huggingface-cli repo create email-triage-env --type space --space-sdk docker

# 3. Clone, copy files, push
git clone https://huggingface.co/spaces/YOUR_USER/email-triage-env hf-space
cp Dockerfile hf-space/
cp -r email_triage_env/ hf-space/
cd hf-space && git add . && git commit -m "Deploy" && git push

# 4. Verify
curl https://YOUR_USER-email-triage-env.hf.space/health

# 5. Run inference
ENV_URL="https://YOUR_USER-email-triage-env.hf.space" python inference.py
```

See `DEPLOYMENT_AND_TESTING_GUIDE.txt` for detailed step-by-step instructions.

---

## 14. Design Decisions & Tradeoffs

### Why 5 emails per task instead of 3?

More emails = more signal for the grader. With only 3 emails, a single
mistake has outsized impact on the average. 5 emails provides smoother
scoring curves and makes trajectory bonuses more meaningful.

### Why deterministic grading instead of LLM-as-judge?

- Reproducible scores across runs (same input = same output)
- No API costs for grading
- No latency overhead
- No dependency on external services
- Easier to debug and validate

### Why WebSocket for inference instead of HTTP?

The HTTP endpoints are stateless — each request creates and destroys a new
environment. This means reset() and step() can't share state. WebSocket
maintains a persistent session, which is required for multi-step episodes.

### Why keyword matching for reply quality?

Semantic similarity (embeddings) would be more sophisticated but adds:
- Heavy dependencies (sentence-transformers, torch)
- Memory requirements (model loading)
- Non-determinism (model versions)
- Slower grading

Keyword matching is fast, deterministic, and sufficient for evaluating
whether the agent addressed the core concern.

### Why trajectory bonuses?

Pure per-email averaging doesn't reward consistency. An agent that scores
[0.9, 0.1, 0.9, 0.1, 0.9] averages 0.56 — same as one scoring
[0.56, 0.56, 0.56, 0.56, 0.56]. But the second agent is clearly more
reliable. The consistency bonus (+0.05) rewards this.

### Why related category partial credit?

Misclassifying a bug as a feature request is less wrong than misclassifying
it as spam. The related-pair system (0.08 partial credit) reflects this
real-world nuance.

---

*End of documentation.*
