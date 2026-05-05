"""Edge-trading simulation against Polymarket.

Strategy
--------
For every market where ``noaa_prob - poly_prob > spread`` we buy YES at
``poly_prob``; if instead ``poly_prob - noaa_prob > spread`` we buy NO at
``1 - poly_prob``. NOAA is treated as the "true" probability for sizing.

Fees are charged on the notional traded (entry + exit assumed). A 1/4 Kelly
sizing rule caps single-trade risk.

Significance
------------
Per-trade PnL (as a fraction of bankroll) is tested against zero with a
one-sided t-test and a percentile bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def kelly_fraction(p_true: float, price: float) -> float:
    """Full-Kelly stake for a binary contract priced at ``price`` paying $1.

    A YES share at price ``price`` returns ``(1 - price)/price`` per dollar
    on win and -1 on loss. The Kelly formula reduces to:

        f* = (p_true - price) / (1 - price)

    Negative values mean "do not bet".
    """
    if price <= 0 or price >= 1:
        return 0.0
    f = (p_true - price) / (1.0 - price)
    return max(f, 0.0)


@dataclass(frozen=True)
class SimulationResult:
    trades: pd.DataFrame
    summary: dict
    bootstrap_ci: tuple[float, float]


def simulate_edge_strategy(
    df: pd.DataFrame,
    *,
    spread_threshold: float = 0.15,
    fee_rate: float = 0.02,
    kelly_fraction_factor: float = 0.25,
    bankroll: float = 1.0,
    rng_seed: int = 7,
) -> SimulationResult:
    """Simulate the (NOAA > Polymarket + spread) strategy.

    Returns a SimulationResult with per-trade PnL, summary stats, and a
    bootstrap CI on mean per-trade ROI.
    """
    needed = {"poly_prob", "noaa_prob", "outcome"}
    if not needed.issubset(df.columns):
        raise ValueError(f"missing columns: {needed - set(df.columns)}")

    rows = []
    for _, r in df.dropna(subset=list(needed)).iterrows():
        poly = float(r["poly_prob"])
        noaa = float(r["noaa_prob"])
        outcome = int(r["outcome"])

        side = None
        if noaa - poly > spread_threshold:
            side, price, p_true, win = "YES", poly, noaa, outcome == 1
        elif poly - noaa > spread_threshold:
            side, price, p_true, win = "NO", 1.0 - poly, 1.0 - noaa, outcome == 0
        if side is None:
            continue

        f_full = kelly_fraction(p_true, price)
        stake = bankroll * f_full * kelly_fraction_factor
        if stake <= 0:
            continue

        gross_pnl = stake * (1.0 - price) / price if win else -stake
        fees = fee_rate * stake  # entry; YES exits at $1 (no extra fee assumed)
        net_pnl = gross_pnl - fees

        rows.append({
            "market_id": r.get("market_id"),
            "city": r.get("city"),
            "title": r.get("title"),
            "poly_prob": poly,
            "noaa_prob": noaa,
            "side": side,
            "entry_price": price,
            "stake": stake,
            "win": int(win),
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "roi": net_pnl / stake,
        })

    trades = pd.DataFrame(rows)
    if trades.empty:
        return SimulationResult(trades, _empty_summary(), (float("nan"), float("nan")))

    rois = trades["roi"].to_numpy()
    t_stat, p_one_sided = _one_sided_t(rois)
    ci_low, ci_high = _bootstrap_mean_ci(rois, rng_seed=rng_seed)

    summary = {
        "n_trades": int(len(trades)),
        "win_rate": float(trades["win"].mean()),
        "gross_roi": float(trades["gross_pnl"].sum() / trades["stake"].sum()),
        "net_roi": float(trades["net_pnl"].sum() / trades["stake"].sum()),
        "mean_trade_roi": float(np.mean(rois)),
        "std_trade_roi": float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0,
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_one_sided),
        "significant_at_5pct": bool(p_one_sided < 0.05),
    }
    return SimulationResult(trades, summary, (ci_low, ci_high))


def _one_sided_t(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    t, p_two = stats.ttest_1samp(x, 0.0)
    p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
    return float(t), float(p_one)


def _bootstrap_mean_ci(
    x: np.ndarray, *, n_boot: int = 5000, alpha: float = 0.05, rng_seed: int = 7
) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def _empty_summary() -> dict:
    return {
        "n_trades": 0,
        "win_rate": float("nan"),
        "gross_roi": float("nan"),
        "net_roi": float("nan"),
        "mean_trade_roi": float("nan"),
        "std_trade_roi": float("nan"),
        "t_statistic": float("nan"),
        "p_value_one_sided": float("nan"),
        "significant_at_5pct": False,
    }
