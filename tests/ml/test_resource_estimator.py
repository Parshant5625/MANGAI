from ml.reserve.resource_estimator import estimate_resource_potential


def test_resource_potential_is_positive_and_ordered():
    result = estimate_resource_potential(0.8, 6.0, seed=7)
    assert result.expected_tonnage > 0
    assert result.p10 <= result.p50 <= result.p90


def test_zero_probability_stays_near_zero():
    result = estimate_resource_potential(0.01, 1.0, probability_std=0.001, seed=1)
    assert result.p90 < result.expected_tonnage * 5
