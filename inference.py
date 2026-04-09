"""
Inference Script — Email Triage Environment
============================================

Uses OpenAI Client for all LLM calls via injected env vars.
Runs all 3 tasks (easy, medium, hard) and prints reproducible scores.
"""

import json
import os
import re
import sys
import time
from typing import Dict, Any, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — use os.environ[] for required vars to fail loudly if missing
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Read API_KEY first (evaluator injects this), fall back to HF_TOKEN for local
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
ENV_NAME = "email_triage_env"

MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 600

# Initialize OpenAI client with the injected proxy URL and key
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# Log config for debugging
print(f"# Config: API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"# Config: MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"# Config: ENV_URL={ENV_URL}", file=sys.stderr)
print(f"# Config: API_KEY={'set' if API_KEY else 'NOT SET'}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert email triage agent for a software company's support team.

For each email you receive, you must respond with a JSON object (and nothing else) with these exact fields:
{
  "priority": one of "urgent", "high", "medium", "low",
  "category": one of "bug_report", "feature_request", "billing", "account_access", "general_inquiry", "spam",
  "reply": "your draft reply to the sender (1-3 sentences, professional)",
  "escalate": true or false
}

Guidelines:
- urgent: production outages, data loss, security breaches
- high: blocking issues, billing disputes, time-sensitive
- medium: general bugs, partnership inquiries
- low: feature requests, spam, nice-to-haves
- escalate: true only for legal threats, VIP customers, data loss, partnership opportunities
- For spam: set priority=low, category=spam, escalate=false, reply should note it's spam
- Reply must be helpful, professional, and address the sender's specific concern
"""


def build_user_message(obs: Dict[str, Any]) -> str:
    parts = []
    if obs.get("feedback"):
        parts.append(f"[Feedback] {obs['feedback']}")
    if obs.get("history"):
        parts.append("[Thread History]\n" + "\n".join(obs["history"]))
    parts.append(f"From: {obs.get('email_from', 'unknown')}")
    parts.append(f"Subject: {obs.get('email_subject', '(no subject)')}")
    parts.append(f"Date: {obs.get('email_timestamp', '')}")
    parts.append(f"Body:\n{obs.get('email_body', '')}")
    parts.append(f"\nEmails remaining: {obs.get('emails_remaining', 0)}")
    return "\n".join(parts)


def parse_llm_response(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response: {text[:200]}")


VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_CATEGORIES = {
    "bug_report", "feature_request", "billing",
    "account_access", "general_inquiry", "spam",
}


def sanitize_action(parsed: Dict[str, Any]) -> Dict[str, Any]:
    priority = parsed.get("priority", "medium").lower().strip()
    if priority not in VALID_PRIORITIES:
        priority = "medium"
    category = parsed.get("category", "general_inquiry").lower().strip()
    if category not in VALID_CATEGORIES:
        category = "general_inquiry"
    reply = str(parsed.get("reply", "Thank you for your email."))[:2000]
    if len(reply.strip()) == 0:
        reply = "Thank you for your email. We will look into this."
    escalate = bool(parsed.get("escalate", False))
    return {"priority": priority, "category": category, "reply": reply, "escalate": escalate}


def action_to_str(action: Dict[str, Any]) -> str:
    p, c, e = action["priority"], action["category"], action["escalate"]
    r = action["reply"][:60].replace("\n", " ")
    return f"triage(p={p},c={c},esc={e},reply='{r}...')"


# ---------------------------------------------------------------------------
# Environment interaction via WebSocket (stateful sessions)
# ---------------------------------------------------------------------------
import asyncio
import websockets


async def ws_run_task(task_id: str) -> Dict[str, Any]:
    """Run a single task over WebSocket. Returns result dict."""
    ws_url = ENV_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")
    print(f"# Connecting to {ws_url}", file=sys.stderr)

    rewards: List[float] = []
    step_num = 0
    success = False

    try:
        async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            reset_resp = json.loads(await ws.recv())
            envelope = reset_resp.get("data", {})
            obs = envelope.get("observation", {})
            done = envelope.get("done", False)

            print(f"# Reset OK, done={done}, subject={obs.get('email_subject', 'N/A')}", file=sys.stderr)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not done and step_num < MAX_STEPS:
                user_msg = build_user_message(obs)
                messages.append({"role": "user", "content": user_msg})

                # Call LLM via OpenAI client (uses injected API_BASE_URL + API_KEY)
                error_msg = None
                print(f"# Calling LLM for step {step_num + 1}...", file=sys.stderr)
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    llm_text = completion.choices[0].message.content or ""
                    print(f"# LLM responded, length={len(llm_text)}", file=sys.stderr)
                except Exception as e:
                    error_msg = str(e)
                    print(f"# LLM error: {error_msg}", file=sys.stderr)
                    llm_text = '{"priority":"medium","category":"general_inquiry","reply":"Thank you for your email.","escalate":false}'

                messages.append({"role": "assistant", "content": llm_text})

                # Parse and sanitize
                try:
                    parsed = parse_llm_response(llm_text)
                except ValueError as e:
                    if error_msg is None:
                        error_msg = str(e)
                    parsed = {}
                action = sanitize_action(parsed)

                # Step the environment
                await ws.send(json.dumps({"type": "step", "data": action}))
                step_resp = json.loads(await ws.recv())
                envelope = step_resp.get("data", {})
                obs = envelope.get("observation", {})
                reward = envelope.get("reward", 0.0)
                if reward is None:
                    reward = 0.0
                done = envelope.get("done", False)

                step_num += 1
                rewards.append(reward)

                # [STEP]
                action_str = action_to_str(action)
                done_str = "true" if done else "false"
                error_str = error_msg if error_msg else "null"
                print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

            success = True

    except Exception as e:
        print(f"# Exception: {type(e).__name__}: {e}", file=sys.stderr)
        success = False

    # [END] — always emitted
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END]   success={success_str} steps={step_num} rewards={rewards_str}")

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_num,
        "rewards": rewards,
        "final_score": rewards[-1] if rewards else 0.0,
    }


def run_task(task_id: str) -> Dict[str, Any]:
    return asyncio.run(ws_run_task(task_id))


def main():
    task_ids = ["easy_triage", "medium_triage", "hard_triage"]
    results = []
    for tid in task_ids:
        result = run_task(tid)
        results.append(result)

    print()
    for r in results:
        print(f"  {r['task_id']:20s} : {r['final_score']:.4f}")
    scores = [r["final_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':20s} : {avg:.4f}")


if __name__ == "__main__":
    main()
