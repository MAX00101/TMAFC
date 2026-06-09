from tmafc.fc.beta_confidence import beta_confidence


def test_zero_majority_returns_zero():
    assert beta_confidence(0, 0) == 0.0
    assert beta_confidence(0, 5) == 0.0


def test_unanimous_majority_high_confidence():
    assert beta_confidence(10, 0) > 0.95


def test_close_majority_low_confidence():
    c = beta_confidence(6, 5)
    assert 0.3 < c < 0.8


def test_smaller_beta0_increases_confidence():
    base = beta_confidence(4, 2, alpha0=1.0, beta0=1.0)
    boosted = beta_confidence(4, 2, alpha0=1.0, beta0=0.1)
    assert boosted > base
