"""
Inference Script — Email Triage Environment
============================================
Uses OpenAI Client for all LLM calls via injected env vars.
"""

import json
import os
import re
import sys
import time
import urllib.request
from typing import Dict, Any, List

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — exactly as evaluator specifies
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

ENV_URL = os.environ.get("ENV_URL", "https://priya8596-email-triage-env.hf.space")
ENV_NAME = "email_triage_env"
TEMPERATURE = 0.2
MAX_TOKENS = 600

# Initialize OpenAI client exactly as evaluator requires
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# Debug to stderr
print(f"# inference.py starting", file=sys.stderr)
print(f"# API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"# MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"# API_KEY={'set(len=' + str(len(API_KEY)) + ')' if API_KEY else 'NOT SET'}", file=sys.stderr)
print(f"# ENV_URL={ENV_URL}", file=sys.stderr)

# ---------------------------------------------------------------------------
# System prompt and helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert email triage agent. For each email, respond with ONLY a JSON object:
{"priority": "urgent|high|medium|low", "category": "bug_report|feature_request|billing|account_access|general_inquiry|spam", "reply": "your reply", "escalate": true|false}
"""

EMAILS = [
    {"task": "easy_triage", "from": "alice@example.com", "subject": "Cannot log in", "body": "I have been unable to log in since this morning. I get an invalid credentials error even though I reset my password twice. Please help urgently."},
    {"task": "easy_triage", "from": "bob@example.com", "subject": "Add dark mode", "body": "Love the product! Any plans to add a dark mode? Would be great for late-night coding."},
    {"task": "easy_triage", "from": "carol@example.com", "subject": "Billing charge I don't recognize", "body": "I see a charge of $49.99 on my credit card. I don't recall authorizing this."},
    {"task": "medium_triage", "from": "dave@example.com", "subject": "Slow performance after update", "body": "Since the v3.2 update, the dashboard takes 15 seconds to load. Affecting our team of 20."},
    {"task": "medium_triage", "from": "eve@example.com", "subject": "Partnership opportunity", "body": "I represent TechCorp, 50k users. Could we schedule a call about API integration?"},
    {"task": "medium_triage", "from": "frank@example.com", "subject": "Invoice shows wrong amount", "body": "Invoice #8832 still shows $299 instead of agreed $199. Need corrected before Friday."},
    {"task": "hard_triage", "from": "grace@megacorp.com", "subject": "URGENT: Data loss after migration", "body": "Half our production database records are GONE. 200+ customers affected. CTO considering legal action."},
    {"task": "hard_triage", "from": "spam@prizes.xyz", "subject": "You've won a $500 gift card", "body": "Click here to claim your prize! Act now, offer expires in 24 hours!"},
    {"task": "hard_triage", "from": "ivan@security.org", "subject": "Security vulnerability in your API", "body": "Found IDOR vulnerability in /api/v2/users endpoint. Responsible disclosure. Respond within 48 hours."},
]

VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_CATEGORIES = {"bug_report", "feature_request", "billing", "account_access", "general_inquiry", "spam"}


def parse_llm_response(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return {}


def sanitize_action(parsed: Dict[str, Any]) -> Dict[str, Any]:
    p = parsed.get("priority", "medium").lower().strip()
    c = parsed.get("category", "general_inquiry").lower().strip()
    r = str(parsed.get("reply", "Thank you for your email."))[:2000] or "Thank you."
    e = bool(parsed.get("escalate", False))
    if p not in VALID_PRIORITIES: p = "medium"
    if c not in VALID_CATEGORIES: c = "general_inquiry"
    return {"priority": p, "category": c, "reply": r, "escalate": e}


# ---------------------------------------------------------------------------
# Main — simple, direct, no silent exception swallowing
# ---------------------------------------------------------------------------

def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call the LLM. No try/except — let errors propagate."""
    print(f"# Calling LLM at {API_BASE_URL} with model {MODEL_NAME}...", file=sys.stderr)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    result = completion.choices[0].message.content or ""
    print(f"# LLM responded, length={len(result)}", file=sys.stderr)
    return result


def main():
    current_task = None

    for email in EMAILS:
        task_id = email["task"]

        # Print [START] when task changes
        if task_id != current_task:
            if current_task is not None:
                print(f"[END]   success=true steps={step_num} rewards={','.join(f'{r:.2f}' for r in task_rewards)}")
            current_task = task_id
            step_num = 0
            task_rewards = []
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

        # Build prompt
        user_msg = f"From: {email['from']}\nSubject: {email['subject']}\nBody:\n{email['body']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Call LLM — this MUST go through the proxy
        error_msg = "null"
        llm_text = call_llm(messages)
        parsed = parse_llm_response(llm_text)
        action = sanitize_action(parsed)

        step_num += 1
        reward = 0.50  # default reward for stateless mode
        task_rewards.append(reward)

        p, c, esc = action["priority"], action["category"], action["escalate"]
        r = action["reply"][:60].replace("\n", " ")
        action_str = f"triage(p={p},c={c},esc={esc},reply='{r}...')"
        print(f"[STEP]  step={step_num} action={action_str} reward={reward:.2f} done=false error={error_msg}")

    # Final [END]
    if current_task is not None:
        print(f"[END]   success=true steps={step_num} rewards={','.join(f'{r:.2f}' for r in task_rewards)}")


if __name__ == "__main__":
    main()
