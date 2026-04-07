"""
Deterministic grader for email triage actions.

Scoring breakdown per email (total 1.0):
  - Priority match:   0.25
  - Category match:   0.25
  - Escalation match: 0.10
  - Reply keywords:   0.20 (required keyword coverage)
  - Reply quality:    0.20 (professionalism, length, tone)

Task score = average across all emails in the task.
"""

from __future__ import annotations

import re
from typing import List

from ..models import EmailAction
from .tasks import Email

# Words that indicate unprofessional tone in a support reply
NEGATIVE_TONE_WORDS = [
    "stupid", "dumb", "idiot", "ridiculous", "whatever",
    "not my problem", "deal with it", "figure it out",
    "lol", "lmao", "smh", "idk",
]

# Words that indicate professional, helpful tone
POSITIVE_TONE_WORDS = [
    "thank", "apolog", "understand", "help", "assist",
    "looking into", "investigate", "resolve", "follow up",
    "please", "happy to", "glad to", "let me",
    "right away", "priority", "team",
]


def _score_reply_quality(reply: str) -> float:
    """
    Score reply professionalism and quality (0.0-1.0).

    Checks:
      - Minimum length (at least 10 words for a real reply)
      - No negative/unprofessional tone words
      - Presence of positive/professional tone words
      - Not ALL CAPS (shouting)
      - Ends with proper sign-off or punctuation
    """
    words = reply.split()
    word_count = len(words)
    reply_lower = reply.lower()
    score = 0.0

    # Length check (0.0-0.30)
    if word_count >= 20:
        score += 0.30
    elif word_count >= 10:
        score += 0.20
    elif word_count >= 5:
        score += 0.10
    # < 5 words = 0

    # Negative tone penalty (0.0 or -0.20)
    has_negative = any(w in reply_lower for w in NEGATIVE_TONE_WORDS)
    if has_negative:
        score -= 0.20

    # Positive tone bonus (0.0-0.40)
    positive_hits = sum(1 for w in POSITIVE_TONE_WORDS if w in reply_lower)
    positive_ratio = min(positive_hits / 4.0, 1.0)  # cap at 4 hits
    score += 0.40 * positive_ratio

    # ALL CAPS penalty
    if word_count > 3 and reply.upper() == reply:
        score -= 0.15

    # Proper ending bonus (0.0-0.15)
    stripped = reply.strip()
    if stripped and stripped[-1] in ".!?":
        score += 0.10
    # Check for sign-off patterns
    if re.search(r"(regards|best|thank|sincerely|cheers)", reply_lower):
        score += 0.05

    # Repetition penalty — same sentence repeated
    sentences = [s.strip() for s in re.split(r'[.!?]', reply) if s.strip()]
    if len(sentences) > 1 and len(set(sentences)) < len(sentences) * 0.5:
        score -= 0.15

    return max(0.0, min(1.0, score))


def grade_single_email(action: EmailAction, gold: Email) -> float:
    """Grade a single triage action against ground truth. Returns 0.0-1.0."""
    score = 0.0

    # --- Priority (0.25) ---
    priority_order = ["low", "medium", "high", "urgent"]
    pred_idx = priority_order.index(action.priority.value)
    gold_idx = priority_order.index(gold.gold_priority)
    diff = abs(pred_idx - gold_idx)
    if diff == 0:
        score += 0.25
    elif diff == 1:
        score += 0.12  # partial credit for adjacent
    elif diff == 2:
        score += 0.05  # small credit for 2-away
    # else 0

    # --- Category (0.25) ---
    if action.category.value == gold.gold_category:
        score += 0.25
    else:
        # Partial credit for related categories
        related_pairs = {
            frozenset({"bug_report", "feature_request"}),
            frozenset({"billing", "account_access"}),
        }
        pair = frozenset({action.category.value, gold.gold_category})
        if pair in related_pairs:
            score += 0.08

    # --- Escalation (0.10) ---
    if action.escalate == gold.should_escalate:
        score += 0.10

    # --- Reply keywords (0.20) ---
    if gold.reply_must_mention:
        reply_lower = action.reply.lower()
        hits = sum(
            1 for kw in gold.reply_must_mention if kw.lower() in reply_lower
        )
        keyword_ratio = hits / len(gold.reply_must_mention)
        score += 0.20 * keyword_ratio

    # --- Reply quality (0.20) ---
    quality = _score_reply_quality(action.reply)
    score += 0.20 * quality

    return round(min(score, 1.0), 4)
