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
import urllib.request
import urllib.error
from typing import Dict, Any, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "https://priya8596-email-triage-env.hf.space")
ENV_NAME = "email_triage_env"

MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 600

# Initialize OpenAI client with injected proxy URL and key
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

print(f"# API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"# MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"# ENV_URL={ENV_URL}", file=sys.stderr)

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

VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_CATEGORIES = {
    "bug_report", "feature_request", "billing",
    "account_access", "general_inquiry", "spam",
}


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
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{ENV_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"# env_post {path} failed: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Hardcoded email tasks (fallback when env is unreachable)
# ---------------------------------------------------------------------------

FALLBACK_EMAILS = {
    "easy_triage": [
        {"email_from": "[email]", "email_subject": "Cannot log in to my account", "email_body": "Hi Support, I have been unable to log in since this morning. I get an invalid credentials error even though I reset my password twice. My username is user_12345. Please help urgently, I have a deadline today. Thanks, Alice", "email_timestamp": "2025-06-15T09:12:00Z", "history": [], "feedback": "Task: Triage 5 straightforward support emails.", "emails_remaining": 5},
        {"email_from": "[email]", "email_subject": "Add dark mode please", "email_body": "Hey team, Love the product! Any plans to add a dark mode? Would be great for late-night coding sessions. Cheers, Bob", "email_timestamp": "2025-06-15T10:30:00Z", "history": [], "feedback": "", "emails_remaining": 4},
        {"email_from": "[email]", "email_subject": "Billing charge I don't recognize", "email_body": "Hello, I see a charge of $49.99 on my credit card from your company dated June 12. I don't recall authorizing this. Can you clarify what this charge is for? Regards, Carol", "email_timestamp": "2025-06-15T11:45:00Z", "history": [], "feedback": "", "emails_remaining": 3},
    ],
    "medium_triage": [
        {"email_from": "[email]", "email_subject": "Slow performance after update", "email_body": "Hi, Since the v3.2 update yesterday, the dashboard takes about 15 seconds to load. It used to be instant. This is affecting our whole team of 20 people. Can you look into it? Dave", "email_timestamp": "2025-06-15T08:00:00Z", "history": [], "feedback": "Task: Triage 5 ambiguous emails.", "emails_remaining": 5},
        {"email_from": "[email]", "email_subject": "Partnership opportunity", "email_body": "Dear Team, I represent TechCorp and we'd like to explore an integration partnership. We have 50k users who could benefit from your API. Could we schedule a call this week? Best, Eve", "email_timestamp": "2025-06-15T09:30:00Z", "history": [], "feedback": "", "emails_remaining": 4},
        {"email_from": "[email]", "email_subject": "Re: Invoice #8832", "email_body": "Hi, Following up on my previous email from last week. The invoice still shows the wrong amount $299 instead of the agreed $199. I need this corrected before our finance team closes the books on Friday. Thanks, Frank", "email_timestamp": "2025-06-15T14:00:00Z", "history": ["Frank (June 8): Hi, invoice #8832 shows $299 but we agreed on $199.", "Support (June 9): Thanks Frank, we're looking into it."], "feedback": "", "emails_remaining": 3},
    ],
    "hard_triage": [
        {"email_from": "[email]", "email_subject": "URGENT: Data loss after migration", "email_body": "HELP! We just ran the migration script from your docs and half our production database records are GONE. This is a P0 for us, 200+ customers affected. We need someone on a call RIGHT NOW. Our CTO is looped in and considering legal action if this isn't resolved today. Grace, VP Engineering at MegaCorp", "email_timestamp": "2025-06-15T07:00:00Z", "history": [], "feedback": "Task: Triage 5 complex emails.", "emails_remaining": 5},
        {"email_from": "[email]", "email_subject": "Congrats! You've won a $500 gift card", "email_body": "Dear valued customer, You have been selected to receive a $500 Amazon gift card! Click here to claim your prize: http://totally-legit-prizes.xyz Act now, offer expires in 24 hours! Best regards, Rewards Team", "email_timestamp": "2025-06-15T06:00:00Z", "history": [], "feedback": "", "emails_remaining": 4},
        {"email_from": "[email]", "email_subject": "Security vulnerability in your API", "email_body": "Hi Security Team, I'm a security researcher and I've found an IDOR vulnerability in your /api/v2/users/{id}/documents endpoint. By changing the user ID parameter, I can access other users' private documents without authentication. I have not disclosed this publicly. Please respond within 48 hours or I will publish my findings. Ivan, Independent Security Researcher", "email_timestamp": "2025-06-15T08:30:00Z", "history": [], "feedback": "", "emails_remaining": 3},
    ],
}


# ---------------------------------------------------------------------------
# Main runner — tries env first, falls back to hardcoded emails
# ---------------------------------------------------------------------------

def get_emails_for_task(task_id: str) -> List[Dict[str, Any]]:
    """Try to get emails from the live environment, fall back to hardcoded."""
    try:
        reset_resp = env_post("/reset", {"task_id": task_id})
        if reset_resp and reset_resp.get("observation"):
            obs = reset_resp["observation"]
            # For HTTP stateless mode, we only get the first email from reset
            # Return it as a single-email list
            return [obs]
    except Exception as e:
        print(f"# Env unreachable, using fallback emails: {e}", file=sys.stderr)

    return FALLBACK_EMAILS.get(task_id, FALLBACK_EMAILS["easy_triage"])


def run_task(task_id: str) -> Dict[str, Any]:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    rewards: List[float] = []
    step_num = 0
    success = False

    try:
        emails = get_emails_for_task(task_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for obs in emails:
            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            # Call LLM via OpenAI client (uses injected API_BASE_URL + API_KEY)
            error_msg = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                llm_text = completion.choices[0].message.content or ""
            except Exception as e:
                error_msg = str(e)
                llm_text = '{"priority":"medium","category":"general_inquiry","reply":"Thank you for your email.","escalate":false}'

            messages.append({"role": "assistant", "content": llm_text})

            try:
                parsed = parse_llm_response(llm_text)
            except ValueError as e:
                if error_msg is None:
                    error_msg = str(e)
                parsed = {}
            action = sanitize_action(parsed)

            # Try to step the environment (may fail in stateless HTTP mode)
            reward = 0.0
            done = False
            try:
                step_resp = env_post("/step", {"action": action})
                if step_resp:
                    reward = step_resp.get("reward", 0.0) or 0.0
                    done = step_resp.get("done", False)
            except Exception:
                pass

            step_num += 1
            rewards.append(reward)

            action_str = action_to_str(action)
            done_str = "true" if done else "false"
            error_str = error_msg if error_msg else "null"
            print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")

        success = True

    except Exception as e:
        print(f"# Fatal error: {e}", file=sys.stderr)
        success = False

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


def main():
    task_ids = ["easy_triage", "medium_triage", "hard_triage"]
    results = []
    for tid in task_ids:
        results.append(run_task(tid))

    print()
    for r in results:
        print(f"  {r['task_id']:20s} : {r['final_score']:.4f}")
    scores = [r["final_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':20s} : {avg:.4f}")


if __name__ == "__main__":
    main()
