from tmafc.ascc.mle import constrained_mle_alpha_beta


def test_easy_history():
    n_list = [2] * 30
    k_list = [2] * 30
    a, b = constrained_mle_alpha_beta(n_list, k_list)
    assert 0.0 < a <= 1.0
    assert 0.0 < b <= 1.0


def test_hard_history():
    n_list = [10] * 30
    k_list = [6] * 30
    a, b = constrained_mle_alpha_beta(n_list, k_list)
    assert 0.0 < a <= 1.0
    assert 0.0 < b <= 1.0


def test_empty_data_returns_init():
    a, b = constrained_mle_alpha_beta([], [], alpha_init=0.5, beta_init=0.7)
    assert (a, b) == (0.5, 0.7)


def test_bounds_respected():
    n_list = [2] * 5
    k_list = [2] * 5
    a, b = constrained_mle_alpha_beta(n_list, k_list, eps_min=0.1, eps_max=0.5)
    assert 0.1 <= a <= 0.5
    assert 0.1 <= b <= 0.5
