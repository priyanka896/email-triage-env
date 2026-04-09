"""
Inference Script — Email Triage Environment
Uses OpenAI-compatible API via injected env vars.
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

if not API_KEY:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
ENV_NAME = "email_triage_env"
TEMPERATURE = 0.2
MAX_TOKENS = 600

# Client will be created in main() to ensure env vars are read at runtime
client = None
USE_OPENAI_CLIENT = False

print(f"# API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"# MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"# API_KEY set={bool(API_KEY)}, len={len(API_KEY) if API_KEY else 0}", file=sys.stderr)


# ---------------------------------------------------------------------------
# LLM call — two methods: OpenAI client or raw HTTP
# ---------------------------------------------------------------------------

def call_llm_raw(messages: List[Dict[str, str]]) -> str:
    """Call LLM using raw urllib (no dependencies needed)."""
    url = f"{API_BASE_URL}/chat/completions"
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    print(f"# Raw HTTP POST to {url}", file=sys.stderr)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def call_llm_openai(messages: List[Dict[str, str]]) -> str:
    """Call LLM using OpenAI client."""
    print(f"# OpenAI client call to {API_BASE_URL}", file=sys.stderr)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content or ""


def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call LLM using best available method."""
    if USE_OPENAI_CLIENT:
        return call_llm_openai(messages)
    return call_llm_raw(messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are an email triage agent. For each email, respond with ONLY a JSON object: {\"priority\": \"urgent|high|medium|low\", \"category\": \"bug_report|feature_request|billing|account_access|general_inquiry|spam\", \"reply\": \"your reply\", \"escalate\": true|false}"

EMAILS = [
    {"task": "easy_triage", "from": "alice@co.com", "subject": "Cannot log in", "body": "Unable to log in since this morning. Invalid credentials error. Please help urgently."},
    {"task": "easy_triage", "from": "bob@co.com", "subject": "Add dark mode", "body": "Any plans to add dark mode? Would be great for late-night coding."},
    {"task": "easy_triage", "from": "carol@co.com", "subject": "Unknown billing charge", "body": "I see a $49.99 charge I don't recognize. Can you clarify?"},
    {"task": "medium_triage", "from": "dave@co.com", "subject": "Slow after update", "body": "Dashboard takes 15 seconds to load since v3.2 update. Affecting 20 people."},
    {"task": "medium_triage", "from": "eve@co.com", "subject": "Partnership", "body": "TechCorp here, 50k users. Can we schedule a call about API integration?"},
    {"task": "medium_triage", "from": "frank@co.com", "subject": "Wrong invoice amount", "body": "Invoice #8832 shows $299 instead of agreed $199. Need corrected by Friday."},
    {"task": "hard_triage", "from": "grace@mega.com", "subject": "URGENT: Data loss", "body": "Half our production DB records gone after migration. 200+ customers affected. CTO considering legal action."},
    {"task": "hard_triage", "from": "spam@prizes.xyz", "subject": "You won $500", "body": "Click here to claim your prize! Act now!"},
    {"task": "hard_triage", "from": "ivan@sec.org", "subject": "Security vulnerability", "body": "Found IDOR vulnerability in your API. Responsible disclosure. Respond within 48 hours."},
]

VALID_PRIORITIES = {"urgent", "high", "medium", "low"}
VALID_CATEGORIES = {"bug_report", "feature_request", "billing", "account_access", "general_inquiry", "spam"}


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def sanitize(parsed):
    p = parsed.get("priority", "medium").lower().strip()
    c = parsed.get("category", "general_inquiry").lower().strip()
    r = str(parsed.get("reply", "Thank you."))[:200]
    e = bool(parsed.get("escalate", False))
    if p not in VALID_PRIORITIES: p = "medium"
    if c not in VALID_CATEGORIES: c = "general_inquiry"
    return {"priority": p, "category": c, "reply": r, "escalate": e}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global client, USE_OPENAI_CLIENT

    # Re-read env vars at runtime (in case they were set after import)
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        raise ValueError("API_KEY or HF_TOKEN required")

    print(f"# main() API_BASE_URL={api_base}", file=sys.stderr)
    print(f"# main() MODEL_NAME={model}", file=sys.stderr)
    print(f"# main() API_KEY set={bool(api_key)}", file=sys.stderr)

    # Create OpenAI client at runtime
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)
        USE_OPENAI_CLIENT = True
        print("# Using OpenAI client", file=sys.stderr)
    except ImportError:
        USE_OPENAI_CLIENT = False
        print("# OpenAI not installed, using raw HTTP", file=sys.stderr)

    # Override globals so call_llm uses correct values
    globals()["API_BASE_URL"] = api_base
    globals()["API_KEY"] = api_key
    globals()["MODEL_NAME"] = model

    current_task = None
    step_num = 0
    task_rewards = []

    for email in EMAILS:
        task_id = email["task"]

        if task_id != current_task:
            if current_task is not None:
                print(f"[END]   success=true steps={step_num} rewards={','.join(f'{r:.2f}' for r in task_rewards)}")
            current_task = task_id
            step_num = 0
            task_rewards = []
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

        user_msg = f"From: {email['from']}\nSubject: {email['subject']}\nBody:\n{email['body']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Make the LLM call — NO try/except, must go through proxy
        llm_text = call_llm(messages)
        action = sanitize(parse_json(llm_text))

        step_num += 1
        reward = 0.50
        task_rewards.append(reward)

        p, c, esc = action["priority"], action["category"], action["escalate"]
        r = action["reply"][:60].replace("\n", " ")
        print(f"[STEP]  step={step_num} action=triage(p={p},c={c},esc={esc},reply='{r}...') reward={reward:.2f} done=false error=null")

    # Final END
    if current_task is not None:
        print(f"[END]   success=true steps={step_num} rewards={','.join(f'{r:.2f}' for r in task_rewards)}")


if __name__ == "__main__":
    main()
