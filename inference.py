"""
Inference Script — Email Triage Environment
============================================
Mandatory STDOUT format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import re
import sys
from typing import Dict, Any, List, Optional

from openai import OpenAI

# Import environment directly (in-process, no HTTP)
from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from email_triage_env.models import EmailAction, Priority, Category

# ── Environment Variables ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY:
    raise EnvironmentError("API_KEY or HF_TOKEN environment variable is required.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

ENV_NAME = "email_triage_env"
MAX_STEPS = 10


# ── Logging helpers ──

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── LLM + parsing ──

SYSTEM_PROMPT = """You are an email triage agent. For each email, respond with ONLY a JSON object:
{"priority": "urgent|high|medium|low", "category": "bug_report|feature_request|billing|account_access|general_inquiry|spam", "reply": "your professional reply", "escalate": true|false}"""

VALID_P = {"urgent", "high", "medium", "low"}
VALID_C = {"bug_report", "feature_request", "billing", "account_access", "general_inquiry", "spam"}


def parse_action(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            p = data.get("priority", "medium").lower().strip()
            c = data.get("category", "general_inquiry").lower().strip()
            r = str(data.get("reply", "Thank you."))[:2000] or "Thank you."
            e = bool(data.get("escalate", False))
            if p not in VALID_P: p = "medium"
            if c not in VALID_C: c = "general_inquiry"
            return EmailAction(priority=Priority(p), category=Category(c), reply=r, escalate=e)
        except Exception:
            pass
    return EmailAction(priority=Priority.MEDIUM, category=Category.GENERAL_INQUIRY, reply="Thank you for your email.", escalate=False)


# ── Task runner ──

def run_task(env, task_id):
    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        obs = env.reset(task_id=task_id)
        print(f"# reset: done={obs.done}, task={obs.task_id}, remaining={obs.emails_remaining}, _task={env._task is not None}", file=sys.stderr)

        for step in range(1, MAX_STEPS + 1):
            # Build prompt from observation
            prompt = f"Subject: {obs.email_subject}\nBody:\n{obs.email_body}"
            if obs.history:
                prompt = f"Thread History:\n" + "\n".join(obs.history) + "\n\n" + prompt
            if obs.feedback:
                prompt = f"Feedback: {obs.feedback}\n\n" + prompt

            # Call LLM
            last_error = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                )
                llm_text = completion.choices[0].message.content or ""
            except Exception as e:
                last_error = str(e)
                log_step(step=step, action="error", reward=0.01, done=True, error=last_error)
                rewards.append(0.01)
                steps_taken = step
                break

            action = parse_action(llm_text)
            action_str = f"triage(p={action.priority.value},c={action.category.value},esc={action.escalate})"

            # Step environment (in-process)
            obs = env.step(action)
            print(f"# step result: done={obs.done}, reward={obs.reward}, task={obs.task_id}, remaining={obs.emails_remaining}, feedback={obs.feedback[:50] if obs.feedback else 'none'}", file=sys.stderr)
            reward = float(obs.reward) if obs.reward is not None else 0.01
            reward = max(0.01, min(0.99, reward))
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                break

        # Compute task score
        if rewards:
            raw_score = rewards[-1]  # Final reward is the task average
        else:
            raw_score = 0.01
        score = max(0.01, min(0.99, raw_score))
        success = score > 0.5

    except Exception as e:
        score = 0.01
        success = False
        print(f"# Error: {e}", file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    env = EmailTriageEnvironment()
    tasks = ["easy_triage", "medium_triage", "hard_triage"]
    for task_id in tasks:
        run_task(env, task_id)


if __name__ == "__main__":
    main()
