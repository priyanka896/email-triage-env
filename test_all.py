"""
Comprehensive unit tests for the Email Triage OpenEnv environment.
Run: python test_all.py
"""
import sys
sys.path.insert(0, ".")

from email_triage_env.models import EmailAction, EmailObservation, EmailState
from email_triage_env.models import Priority, Category
from email_triage_env.server.email_triage_environment import EmailTriageEnvironment


def test_easy_task():
    env = EmailTriageEnvironment()
    obs = env.reset(task_id="easy_triage")

    assert obs.task_id == "easy_triage"
    assert obs.email_subject == "Cannot log in to my account"
    assert obs.emails_remaining == 5
    assert obs.done is False
    print("[PASS] Easy task reset — 5 emails loaded")

    actions = [
        EmailAction(priority=Priority.URGENT, category=Category.ACCOUNT_ACCESS,
                     reply="I understand you cannot log in. Let me help reset your password and restore account access right away.",
                     escalate=False),
        EmailAction(priority=Priority.LOW, category=Category.FEATURE_REQUEST,
                     reply="Thanks for the feature request! We will add dark mode to our roadmap and follow up.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BILLING,
                     reply="I see the charge on your account. Let me investigate this billing issue for you.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BUG_REPORT,
                     reply="Sorry about the crash. We are investigating this version compatibility issue right away.",
                     escalate=False),
        EmailAction(priority=Priority.LOW, category=Category.GENERAL_INQUIRY,
                     reply="You can export your data from the settings page. Let me help you find the export option.",
                     escalate=False),
    ]

    for i, action in enumerate(actions):
        obs = env.step(action)
        if i < 4:
            assert obs.done is False
        print(f"[PASS] Easy email {i+1}: reward={obs.reward}")

    assert obs.done is True
    assert obs.reward > 0.85, f"Perfect answers should score high, got {obs.reward}"
    print(f"[PASS] Easy task complete: final={obs.reward:.4f}")


def test_medium_task():
    env = EmailTriageEnvironment()
    obs = env.reset(task_id="medium_triage")

    assert obs.task_id == "medium_triage"
    assert obs.emails_remaining == 5
    print("[PASS] Medium task reset — 5 emails loaded")

    actions = [
        EmailAction(priority=Priority.HIGH, category=Category.BUG_REPORT,
                     reply="We are investigating the performance issue after the update. Our team is looking into it.",
                     escalate=False),
        EmailAction(priority=Priority.MEDIUM, category=Category.GENERAL_INQUIRY,
                     reply="Thank you for the partnership opportunity. Let me schedule a call with our team.",
                     escalate=True),
        EmailAction(priority=Priority.HIGH, category=Category.BILLING,
                     reply="I apologize for the delay. The invoice amount will be corrected to the agreed amount.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BUG_REPORT,
                     reply="We are investigating the webhook issue. It may be related to our recent maintenance window.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BILLING,
                     reply="I understand your frustration. Let me process your cancellation and refund for the subscription immediately.",
                     escalate=True),
    ]

    for i, action in enumerate(actions):
        obs = env.step(action)
        print(f"[PASS] Medium email {i+1}: reward={obs.reward}")

    assert obs.done is True
    assert obs.reward > 0.7, f"Correct answers should score well, got {obs.reward}"
    print(f"[PASS] Medium task complete: final={obs.reward:.4f}")


def test_hard_task():
    env = EmailTriageEnvironment()
    obs = env.reset(task_id="hard_triage")

    assert obs.task_id == "hard_triage"
    assert obs.emails_remaining == 5
    print("[PASS] Hard task reset — 5 emails loaded")

    actions = [
        EmailAction(priority=Priority.URGENT, category=Category.BUG_REPORT,
                     reply="This is being escalated immediately. Our team will provide immediate assistance with the data loss issue.",
                     escalate=True),
        EmailAction(priority=Priority.LOW, category=Category.SPAM,
                     reply="This email has been identified as spam and will be discarded. No action needed.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BUG_REPORT,
                     reply="We are fixing the CSV export UTF-8 encoding issue for non-English characters. Thank you for reporting.",
                     escalate=False),
        EmailAction(priority=Priority.URGENT, category=Category.BUG_REPORT,
                     reply="Thank you for the responsible security disclosure. This is being escalated to our security team immediately.",
                     escalate=True),
        EmailAction(priority=Priority.URGENT, category=Category.BILLING,
                     reply="I sincerely apologize for the delay. Let me upgrade your account immediately and investigate what happened.",
                     escalate=True),
    ]

    for i, action in enumerate(actions):
        obs = env.step(action)
        print(f"[PASS] Hard email {i+1}: reward={obs.reward}")

    assert obs.done is True
    assert obs.reward > 0.7, f"Correct answers should score well, got {obs.reward}"
    print(f"[PASS] Hard task complete: final={obs.reward:.4f}")


def test_partial_scoring():
    """Verify partial credit works correctly."""
    env = EmailTriageEnvironment()
    env.reset(task_id="easy_triage")

    # Completely wrong answer
    obs = env.step(EmailAction(
        priority=Priority.LOW, category=Category.SPAM,
        reply="ok",
        escalate=True,
    ))
    assert obs.reward < 0.15, f"Wrong answer should score very low, got {obs.reward}"
    print(f"[PASS] Partial scoring — wrong answer: reward={obs.reward}")

    # Adjacent priority (medium instead of low for feature request)
    obs = env.step(EmailAction(
        priority=Priority.MEDIUM, category=Category.FEATURE_REQUEST,
        reply="Thanks for the feature request! We appreciate your feedback.",
        escalate=False,
    ))
    assert 0.4 < obs.reward < 1.0, f"Adjacent should get partial, got {obs.reward}"
    print(f"[PASS] Partial scoring — adjacent priority: reward={obs.reward}")


def test_reply_quality_grading():
    """Verify the reply quality grader works."""
    from email_triage_env.server.grader import _score_reply_quality

    # Good professional reply
    good = _score_reply_quality(
        "Thank you for reaching out. I understand your concern and our team "
        "is investigating this issue. We will follow up with you shortly."
    )
    assert good > 0.6, f"Good reply should score high, got {good}"
    print(f"[PASS] Reply quality — good reply: {good:.4f}")

    # Terrible reply
    bad = _score_reply_quality("ok")
    assert bad < 0.2, f"Bad reply should score low, got {bad}"
    print(f"[PASS] Reply quality — bad reply: {bad:.4f}")

    # ALL CAPS reply
    caps = _score_reply_quality("I WILL LOOK INTO THIS FOR YOU RIGHT AWAY")
    assert caps < 0.5, f"ALL CAPS should be penalized, got {caps}"
    print(f"[PASS] Reply quality — ALL CAPS penalty: {caps:.4f}")

    # Unprofessional reply
    rude = _score_reply_quality("Whatever, just deal with it yourself. Not my problem lol.")
    assert rude < 0.4, f"Rude reply should score low, got {rude}"
    print(f"[PASS] Reply quality — unprofessional penalty: {rude:.4f}")


def test_related_category_partial_credit():
    """Verify related categories get partial credit."""
    from email_triage_env.server.grader import grade_single_email
    from email_triage_env.server.tasks import Email

    gold = Email(
        sender="test@test.com", subject="Test", body="Test",
        timestamp="2025-01-01T00:00:00Z",
        gold_priority="high", gold_category="bug_report",
        should_escalate=False, reply_must_mention=[],
    )

    # feature_request is related to bug_report
    action = EmailAction(
        priority=Priority.HIGH, category=Category.FEATURE_REQUEST,
        reply="Thank you for your feedback. We are looking into this issue.",
        escalate=False,
    )
    score = grade_single_email(action, gold)

    # Should get: priority(0.25) + related_cat(0.08) + escalation(0.10) + quality(~0.15)
    assert score > 0.4, f"Related category should get partial credit, got {score}"
    print(f"[PASS] Related category partial credit: {score:.4f}")


def test_trajectory_bonus():
    """Verify trajectory bonuses are applied."""
    env = EmailTriageEnvironment()
    env.reset(task_id="easy_triage")

    # Give consistently good answers to trigger consistency bonus
    good_actions = [
        EmailAction(priority=Priority.URGENT, category=Category.ACCOUNT_ACCESS,
                     reply="Let me help you reset your password and restore account access right away.",
                     escalate=False),
        EmailAction(priority=Priority.LOW, category=Category.FEATURE_REQUEST,
                     reply="Thanks for the feature request! We will add this to our roadmap.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BILLING,
                     reply="I see the charge on your account. Let me investigate this billing issue.",
                     escalate=False),
        EmailAction(priority=Priority.HIGH, category=Category.BUG_REPORT,
                     reply="Sorry about the crash. We are investigating this version issue right away.",
                     escalate=False),
        EmailAction(priority=Priority.LOW, category=Category.GENERAL_INQUIRY,
                     reply="You can export your data from the settings page. Let me help you.",
                     escalate=False),
    ]

    for action in good_actions:
        obs = env.step(action)

    assert obs.done is True
    # With trajectory bonus, should be higher than raw average
    assert "bonus" in obs.feedback.lower()
    print(f"[PASS] Trajectory bonus applied: final={obs.reward:.4f}")
    print(f"       Feedback: {obs.feedback}")


def test_state_tracking():
    """Verify state tracking."""
    env = EmailTriageEnvironment()
    env.reset(task_id="easy_triage")

    assert env.state.step_count == 0
    assert env.state.current_email_index == 0
    assert env.state.total_emails == 5

    env.step(EmailAction(
        priority=Priority.URGENT, category=Category.ACCOUNT_ACCESS,
        reply="Resetting your password and account access now. Please try again.",
        escalate=False,
    ))

    assert env.state.step_count == 1
    assert env.state.current_email_index == 1
    assert env.state.cumulative_reward > 0
    print(f"[PASS] State tracking: steps={env.state.step_count}, cumulative={env.state.cumulative_reward:.4f}")


def test_reset_clears_state():
    """Verify reset clears previous episode."""
    env = EmailTriageEnvironment()
    env.reset(task_id="easy_triage")
    env.step(EmailAction(
        priority=Priority.URGENT, category=Category.ACCOUNT_ACCESS,
        reply="Resetting your password now. Please try again shortly.",
        escalate=False,
    ))
    assert env.state.step_count == 1

    obs = env.reset(task_id="medium_triage")
    assert env.state.step_count == 0
    assert env.state.current_email_index == 0
    assert obs.task_id == "medium_triage"
    print("[PASS] Reset clears state correctly")


def test_invalid_task_fallback():
    """Verify invalid task_id falls back to easy."""
    env = EmailTriageEnvironment()
    obs = env.reset(task_id="nonexistent_task")
    assert obs.task_id == "easy_triage"
    print("[PASS] Invalid task_id falls back to easy_triage")


def test_step_without_reset():
    """Verify stepping without reset returns error observation."""
    env = EmailTriageEnvironment()
    obs = env.step(EmailAction(
        priority=Priority.MEDIUM, category=Category.GENERAL_INQUIRY,
        reply="Hello, thank you for reaching out.",
        escalate=False,
    ))
    assert obs.done is True
    assert "not initialized" in obs.feedback.lower() or "reset" in obs.feedback.lower()
    print("[PASS] Step without reset returns error gracefully")


def test_score_range():
    """Verify all scores are in 0.0-1.0 range across all tasks."""
    for task_id in ["easy_triage", "medium_triage", "hard_triage"]:
        env = EmailTriageEnvironment()
        env.reset(task_id=task_id)

        for _ in range(5):
            obs = env.step(EmailAction(
                priority=Priority.MEDIUM, category=Category.GENERAL_INQUIRY,
                reply="Thank you for your email. We will look into this matter and get back to you.",
                escalate=False,
            ))
            reward = obs.reward
            assert 0.0 <= reward <= 1.0, f"Score {reward} out of range for {task_id}"

    print("[PASS] All scores in 0.0-1.0 range across all tasks")


def test_metadata():
    """Verify environment metadata."""
    env = EmailTriageEnvironment()
    meta = env.get_metadata()
    assert meta.name == "Email Triage Environment"
    assert "email" in meta.description.lower()
    assert meta.version == "0.2.0"
    print(f"[PASS] Metadata: name={meta.name}, version={meta.version}")


def test_grader_components():
    """Test individual grading components."""
    from email_triage_env.server.grader import grade_single_email
    from email_triage_env.server.tasks import Email

    gold = Email(
        sender="test@test.com", subject="Test", body="Test body",
        timestamp="2025-01-01T00:00:00Z",
        gold_priority="urgent", gold_category="bug_report",
        should_escalate=True, reply_must_mention=["bug", "fix"],
    )

    # Perfect answer
    perfect = EmailAction(
        priority=Priority.URGENT, category=Category.BUG_REPORT,
        reply="We found the bug and will fix it immediately. Thank you for reporting this issue. Our team is investigating and we will follow up shortly.",
        escalate=True,
    )
    score = grade_single_email(perfect, gold)
    assert score > 0.85, f"Perfect answer should be >0.85, got {score}"
    print(f"[PASS] Grader — perfect answer: {score}")

    # Only priority correct
    only_priority = EmailAction(
        priority=Priority.URGENT, category=Category.SPAM,
        reply="Thanks for reaching out.",
        escalate=False,
    )
    score = grade_single_email(only_priority, gold)
    assert 0.2 <= score <= 0.5, f"Only priority correct should be moderate, got {score}"
    print(f"[PASS] Grader — only priority correct: {score}")

    # All wrong
    wrong = EmailAction(
        priority=Priority.LOW, category=Category.SPAM,
        reply="ok",
        escalate=False,
    )
    score = grade_single_email(wrong, gold)
    assert score < 0.1, f"All wrong should be near 0, got {score}"
    print(f"[PASS] Grader — all wrong: {score}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EMAIL TRIAGE ENVIRONMENT — COMPREHENSIVE TESTS")
    print("=" * 60)
    print()

    tests = [
        ("Easy Task (5 emails, correct answers)", test_easy_task),
        ("Medium Task (5 emails, correct answers)", test_medium_task),
        ("Hard Task (5 emails, correct answers)", test_hard_task),
        ("Partial Scoring", test_partial_scoring),
        ("Reply Quality Grading", test_reply_quality_grading),
        ("Related Category Partial Credit", test_related_category_partial_credit),
        ("Trajectory Bonus", test_trajectory_bonus),
        ("State Tracking", test_state_tracking),
        ("Reset Clears State", test_reset_clears_state),
        ("Invalid Task Fallback", test_invalid_task_fallback),
        ("Step Without Reset", test_step_without_reset),
        ("Score Range Validation", test_score_range),
        ("Environment Metadata", test_metadata),
        ("Grader Components", test_grader_components),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
