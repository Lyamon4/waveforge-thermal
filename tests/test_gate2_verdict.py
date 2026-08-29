"""Tests for machine-readable Gate 2A outcome semantics."""

from waveforge.verification.compare import Gate2Status, classify_campaign


def test_valid_campaign_below_effect_is_no_go_not_invalid() -> None:
    """A valid negative result must not be classified as a numerical failure."""
    verdict = classify_campaign(valid=True, passing_seed_count=1, required=2)

    assert verdict.status is Gate2Status.NO_GO_EFFECT
    assert verdict.reason_codes == ("INSUFFICIENT_PASSING_SEEDS",)


def test_numerical_failure_is_invalid_run() -> None:
    """Numerical invalidity must take precedence over apparent effect."""
    verdict = classify_campaign(valid=False, passing_seed_count=3, required=2)

    assert verdict.status is Gate2Status.INVALID_RUN
    assert verdict.reason_codes == ("NUMERICAL_OR_INTEGRITY_FAILURE",)


def test_valid_campaign_with_required_seeds_passes() -> None:
    """A valid campaign meeting the locked seed count must pass this contract."""
    verdict = classify_campaign(valid=True, passing_seed_count=2, required=2)

    assert verdict.status is Gate2Status.PASS
    assert verdict.reason_codes == ()
