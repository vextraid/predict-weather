import numpy as np
import pandas as pd

from predict_weather.simulation import (
    kelly_fraction,
    simulate_edge_strategy,
)


def test_kelly_no_edge():
    assert kelly_fraction(0.4, 0.5) == 0.0


def test_kelly_basic():
    f = kelly_fraction(0.6, 0.5)
    assert abs(f - 0.2) < 1e-9


def test_simulation_no_trades_when_spread_too_small():
    df = pd.DataFrame({
        "poly_prob": [0.5, 0.5],
        "noaa_prob": [0.55, 0.55],
        "outcome":   [1, 0],
    })
    res = simulate_edge_strategy(df, spread_threshold=0.15)
    assert res.summary["n_trades"] == 0


def test_simulation_takes_yes_bets_when_noaa_higher():
    rng = np.random.default_rng(0)
    n = 400
    poly = np.full(n, 0.30)
    noaa = np.full(n, 0.55)
    outcomes = (rng.random(n) < noaa).astype(int)
    df = pd.DataFrame({"poly_prob": poly, "noaa_prob": noaa, "outcome": outcomes})

    res = simulate_edge_strategy(df, spread_threshold=0.15, fee_rate=0.0)
    assert res.summary["n_trades"] == n
    # NOAA-true → strategy should be profitable in expectation.
    assert res.summary["net_roi"] > 0
    assert res.summary["p_value_one_sided"] < 0.05


def test_simulation_takes_no_bets_when_polymarket_higher():
    rng = np.random.default_rng(1)
    n = 400
    poly = np.full(n, 0.80)
    noaa = np.full(n, 0.50)
    outcomes = (rng.random(n) < noaa).astype(int)
    df = pd.DataFrame({"poly_prob": poly, "noaa_prob": noaa, "outcome": outcomes})

    res = simulate_edge_strategy(df, spread_threshold=0.15, fee_rate=0.0)
    assert res.summary["n_trades"] == n
    assert (res.trades["side"] == "NO").all()
