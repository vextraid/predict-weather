import numpy as np
import pandas as pd

from predict_weather.calibration import brier_score, calibration_table


def test_brier_perfect():
    p = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0, 1, 0, 1])
    assert brier_score(p, y) == 0.0


def test_brier_worst():
    p = np.array([1.0, 0.0])
    y = np.array([0, 1])
    assert brier_score(p, y) == 1.0


def test_brier_handles_nans():
    p = np.array([np.nan, 0.5, 0.5])
    y = np.array([1, 1, 0])
    assert brier_score(p, y) == 0.25


def test_calibration_table_flags_large_gap():
    df = pd.DataFrame({
        "poly_prob": [0.05, 0.05, 0.05, 0.55, 0.55],
        "noaa_prob": [0.50, 0.50, 0.50, 0.55, 0.55],
        "outcome":   [   0,    0,    0,    1,    0],
    })
    cal = calibration_table(df)
    assert cal["flagged"].any()
    bin0 = cal[cal["bin_left"] <= 0.05].iloc[0]
    assert bin0["abs_gap"] > 0.15
