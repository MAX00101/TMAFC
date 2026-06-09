from tmafc.fc.fc_runner import FCRunner


def test_fc_converges_on_unanimous_samples():
    calls = []

    def sample_fn(states):
        calls.append(list(states.keys()))
        return {cid: "Genesis" for cid in states.keys()}

    fc = FCRunner(tau=0.9, n_max=10, alpha0=1.0, beta0=0.5, min_samples_for_conf=2)
    result = fc.run(["brand"], sample_fn)
    assert result.converged_values["brand"] == "Genesis"
    assert result.samples_consumed <= 10
    assert all("brand" in c for c in calls)


def test_fc_drops_converged_components_from_pending():
    counts = {"fast": 0, "slow": 0}

    def sample_fn(states):
        out = {}
        if "fast" in states:
            counts["fast"] += 1
            out["fast"] = "A"
        if "slow" in states:
            counts["slow"] += 1
            out["slow"] = "B" if counts["slow"] % 2 == 0 else "C"
        return out

    fc = FCRunner(tau=0.85, n_max=20, beta0=0.3)
    result = fc.run(["fast", "slow"], sample_fn)
    assert counts["fast"] <= counts["slow"]
    assert result.converged_values["fast"] == "A"


def test_fc_respects_n_max():
    def sample_fn(states):
        return {cid: "X" if hash(cid) % 2 else "Y" for cid in states}

    fc = FCRunner(tau=0.99, n_max=5, beta0=1.0)
    result = fc.run(["k1"], sample_fn)
    assert result.samples_consumed <= 5
