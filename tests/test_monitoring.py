import pandas as pd

from src.data.generate_data import generate
from src.monitoring.drift import population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    values = pd.Series([1, 2, 3, 4, 5] * 20)
    assert population_stability_index(values, values) < 0.001


def test_psi_detects_shifted_distribution():
    baseline = pd.Series(range(1, 101))
    incoming = pd.Series(range(101, 201))
    assert population_stability_index(baseline, incoming) > 0.25
