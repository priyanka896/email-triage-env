"""
Concrete email datasets for each task.
"""

from .tasks import Email, Task


def get_easy_task() -> Task:
    return Task(
        task_id="easy_triage",
        description="Triage 5 straightforward support emails with clear signals.",
        difficulty="easy",
        emails=[
            Email(
                sender="[email]",
                subject="Cannot log in to my account",
                body=(
                    "Hi Support,\n\n"
                    "I have been unable to log in since this morning. "
                    "I get an 'invalid credentials' error even though I reset "
                    "my password twice. My username is user_12345. "
                    "Please help urgently, I have a deadline today.\n\n"
                    "Thanks,\nAlice"
                ),
                timestamp="2025-06-15T09:12:00Z",
                gold_priority="urgent",
                gold_category="account_access",
                should_escalate=False,
                reply_must_mention=["password", "account", "reset"],
            ),
            Email(
                sender="[email]",
                subject="Add dark mode please",
                body=(
                    "Hey team,\n\n"
                    "Love the product! Any plans to add a dark mode? "
                    "Would be great for late-night coding sessions.\n\n"
                    "Cheers,\nBob"
                ),
                timestamp="2025-06-15T10:30:00Z",
                gold_priority="low",
                gold_category="feature_request",
                should_escalate=False,
                reply_must_mention=["feature", "request"],
            ),
            Email(
                sender="[email]",
                subject="Billing charge I don't recognize",
                body=(
                    "Hello,\n\n"
                    "I see a charge of $49.99 on my credit card from your company "
                    "dated June 12. I don't recall authorizing this. "
                    "Can you clarify what this charge is for?\n\n"
                    "Regards,\nCarol"
                ),
                timestamp="2025-06-15T11:45:00Z",
                gold_priority="high",
                gold_category="billing",
                should_escalate=False,
                reply_must_mention=["charge", "billing"],
            ),
            Email(
                sender="[email]",
                subject="App crashes on startup",
                body=(
                    "Hi,\n\n"
                    "Every time I open the mobile app it crashes immediately. "
                    "I'm on iOS 18.1, app version 4.2.0. "
                    "I've tried reinstalling but the problem persists.\n\n"
                    "Thanks,\nDiana"
                ),
                timestamp="2025-06-15T12:15:00Z",
                gold_priority="high",
                gold_category="bug_report",
                should_escalate=False,
                reply_must_mention=["crash", "version", "investigating"],
            ),
            Email(
                sender="[email]",
                subject="How do I export my data?",
                body=(
                    "Hello,\n\n"
                    "I'd like to download all my project data as a backup. "
                    "Is there an export option somewhere? I looked in settings "
                    "but couldn't find it.\n\n"
                    "Thanks,\nEdward"
                ),
                timestamp="2025-06-15T13:00:00Z",
                gold_priority="low",
                gold_category="general_inquiry",
                should_escalate=False,
                reply_must_mention=["export", "data", "settings"],
            ),
        ],
    )


def get_medium_task() -> Task:
    return Task(
        task_id="medium_triage",
        description=(
            "Triage 5 ambiguous emails where priority and category "
            "require careful reading."
        ),
        difficulty="medium",
        emails=[
            Email(
                sender="[email]",
                subject="Slow performance after update",
                body=(
                    "Hi,\n\n"
                    "Since the v3.2 update yesterday, the dashboard takes "
                    "about 15 seconds to load. It used to be instant. "
                    "This is affecting our whole team of 20 people. "
                    "We rely on this for daily standups.\n\n"
                    "Can you look into it?\n\nDave"
                ),
                timestamp="2025-06-15T08:00:00Z",
                gold_priority="high",
                gold_category="bug_report",
                should_escalate=False,
                reply_must_mention=["performance", "update", "investigating"],
            ),
            Email(
                sender="[email]",
                subject="Partnership opportunity",
                body=(
                    "Dear Team,\n\n"
                    "I represent TechCorp and we'd like to explore an "
                    "integration partnership. We have 50k users who could "
                    "benefit from your API. Could we schedule a call this week?\n\n"
                    "Best,\nEve"
                ),
                timestamp="2025-06-15T09:30:00Z",
                gold_priority="medium",
                gold_category="general_inquiry",
                should_escalate=True,
                reply_must_mention=["partnership", "schedule"],
            ),
            Email(
                sender="[email]",
                subject="Re: Invoice #8832",
                body=(
                    "Hi,\n\n"
                    "Following up on my previous email from last week. "
                    "The invoice still shows the wrong amount — $299 instead "
                    "of the agreed $199. I need this corrected before our "
                    "finance team closes the books on Friday.\n\n"
                    "Thanks,\nFrank"
                ),
                timestamp="2025-06-15T14:00:00Z",
                thread_history=[
                    "Frank (June 8): Hi, invoice #8832 shows $299 but we agreed on $199.",
                    "Support (June 9): Thanks Frank, we're looking into it.",
                ],
                gold_priority="high",
                gold_category="billing",
                should_escalate=False,
                reply_must_mention=["invoice", "corrected", "amount"],
            ),
            Email(
                sender="[email]",
                subject="Webhook notifications stopped working",
                body=(
                    "Hey,\n\n"
                    "Our webhook endpoint hasn't received any events since "
                    "around 2pm yesterday. We checked our server logs and "
                    "there's nothing incoming. Our endpoint is healthy — "
                    "I tested it manually. This is breaking our CI pipeline.\n\n"
                    "Could be related to your maintenance window?\n\nGeorge"
                ),
                timestamp="2025-06-15T10:00:00Z",
                gold_priority="high",
                gold_category="bug_report",
                should_escalate=False,
                reply_must_mention=["webhook", "investigating", "maintenance"],
            ),
            Email(
                sender="[email]",
                subject="Cancellation request — unhappy with service",
                body=(
                    "To whom it may concern,\n\n"
                    "I want to cancel my Pro subscription effective immediately. "
                    "The recent price increase from $29 to $49/month was not "
                    "communicated properly and I feel misled. I also want a "
                    "refund for this month.\n\n"
                    "If this isn't resolved I'll dispute the charge with my bank.\n\n"
                    "Helen"
                ),
                timestamp="2025-06-15T15:30:00Z",
                gold_priority="high",
                gold_category="billing",
                should_escalate=True,
                reply_must_mention=["cancel", "refund", "subscription"],
            ),
        ],
    )


def get_hard_task() -> Task:
    return Task(
        task_id="hard_triage",
        description=(
            "Triage 5 complex emails with conflicting signals, "
            "multi-threaded context, and edge cases."
        ),
        difficulty="hard",
        emails=[
            Email(
                sender="[email]",
                subject="URGENT: Data loss after migration",
                body=(
                    "HELP!\n\n"
                    "We just ran the migration script from your docs and "
                    "half our production database records are GONE. "
                    "This is a P0 for us — 200+ customers affected. "
                    "We need someone on a call RIGHT NOW.\n\n"
                    "Our CTO is looped in and considering legal action "
                    "if this isn't resolved today.\n\n"
                    "— Grace, VP Engineering at MegaCorp"
                ),
                timestamp="2025-06-15T07:00:00Z",
                gold_priority="urgent",
                gold_category="bug_report",
                should_escalate=True,
                reply_must_mention=["data", "escalat", "immediate"],
            ),
            Email(
                sender="[email]",
                subject="Congrats! You've won a $500 gift card",
                body=(
                    "Dear valued customer,\n\n"
                    "You have been selected to receive a $500 Amazon gift card! "
                    "Click here to claim your prize: http://totally-legit-prizes.xyz\n\n"
                    "Act now — offer expires in 24 hours!\n\n"
                    "Best regards,\nRewards Team"
                ),
                timestamp="2025-06-15T06:00:00Z",
                gold_priority="low",
                gold_category="spam",
                should_escalate=False,
                reply_must_mention=["spam"],
            ),
            Email(
                sender="[email]",
                subject="Re: Re: Re: Feature + Bug combo",
                body=(
                    "Hi again,\n\n"
                    "So the export-to-CSV feature I requested last month "
                    "actually shipped — thanks! But it's broken: the exported "
                    "file has garbled UTF-8 characters for any non-English text. "
                    "Our Japanese clients are complaining.\n\n"
                    "Also, while you're at it, could you add PDF export too? "
                    "That would solve the formatting issues entirely.\n\n"
                    "Priority-wise, the CSV bug is blocking us, but the PDF "
                    "thing can wait.\n\n"
                    "Cheers,\nHiro"
                ),
                timestamp="2025-06-15T13:00:00Z",
                thread_history=[
                    "Hiro (May 10): Can we get CSV export for reports?",
                    "Support (May 11): Great idea! We'll add it to the roadmap.",
                    "Hiro (June 1): Any update on CSV export?",
                    "Support (June 5): Shipped in v3.1! Let us know how it works.",
                ],
                gold_priority="high",
                gold_category="bug_report",
                should_escalate=False,
                reply_must_mention=["CSV", "UTF", "encoding"],
            ),
            Email(
                sender="[email]",
                subject="Security vulnerability in your API",
                body=(
                    "Hi Security Team,\n\n"
                    "I'm a security researcher and I've found an IDOR vulnerability "
                    "in your /api/v2/users/{id}/documents endpoint. By changing the "
                    "user ID parameter, I can access other users' private documents "
                    "without authentication.\n\n"
                    "I have not disclosed this publicly. I'd like to follow "
                    "responsible disclosure. Please respond within 48 hours or "
                    "I will publish my findings.\n\n"
                    "PoC and details available upon request.\n\n"
                    "— Ivan, Independent Security Researcher"
                ),
                timestamp="2025-06-15T08:30:00Z",
                gold_priority="urgent",
                gold_category="bug_report",
                should_escalate=True,
                reply_must_mention=["security", "escalat", "disclosure"],
            ),
            Email(
                sender="[email]",
                subject="Re: Account upgrade — still waiting",
                body=(
                    "This is my THIRD email about this.\n\n"
                    "I paid for the Enterprise plan two weeks ago. My account "
                    "still shows Free tier. I've attached the payment receipt "
                    "twice already. Nobody has responded.\n\n"
                    "I'm starting to think this company is a scam. If my account "
                    "isn't upgraded by end of day I'm filing a chargeback and "
                    "leaving a review on every platform I can find.\n\n"
                    "— Julia"
                ),
                timestamp="2025-06-15T16:00:00Z",
                thread_history=[
                    "Julia (June 1): I upgraded to Enterprise but my account still shows Free.",
                    "Julia (June 8): Following up — still no upgrade. Payment was processed.",
                ],
                gold_priority="urgent",
                gold_category="billing",
                should_escalate=True,
                reply_must_mention=["upgrade", "account", "apolog"],
            ),
        ],
    )


TASKS = {
    "easy_triage": get_easy_task,
    "medium_triage": get_medium_task,
    "hard_triage": get_hard_task,
}
