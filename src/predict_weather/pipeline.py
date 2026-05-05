"""End-to-end orchestrator: collect -> analyse -> simulate -> report.

Run modes
---------
* ``--source live``       hits the real Polymarket + NOAA APIs.
* ``--source synthetic``  uses the reproducible synthetic dataset (default
                          fallback when ``live`` raises a network error).

Outputs
-------
* ``output/observations.csv`` — one row per market.
* ``output/calibration_table.csv`` — per-bucket calibration.
* ``output/trades.csv`` — simulated trades.
* ``output/reliability.png`` — reliability diagram.
* ``output/summary.json`` — Brier scores, ROI, p-value, bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .calibration import brier_score, calibration_table, plot_reliability
from .data_model import MarketObservation, to_dataframe
from .noaa import NOAAClient
from .polymarket import PolymarketClient
from .simulation import simulate_edge_strategy
from .synthetic import build_synthetic_dataset

log = logging.getLogger("pipeline")
DEFAULT_OUTPUT = Path("output")


def collect_live(months: int = 6) -> list[MarketObservation]:
    """Collect resolved markets from the live APIs and attach NOAA probs."""
    since = datetime.now(timezone.utc) - timedelta(days=months * 30)
    poly = PolymarketClient()
    noaa = NOAAClient()
    raw = poly.collect_observations(since=since)

    enriched: list[MarketObservation] = []
    for obs in raw:
        try:
            noaa_prob = noaa.probability_for(obs)
        except Exception:
            log.exception("NOAA lookup failed for %s", obs.market_id)
            continue
        enriched.append(MarketObservation(
            market_id=obs.market_id,
            title=obs.title,
            city=obs.city,
            target_date=obs.target_date,
            resolution_at=obs.resolution_at,
            poly_prob=obs.poly_prob,
            noaa_prob=float(noaa_prob),
            outcome=obs.outcome,
        ))
    return enriched


def run(source: str = "synthetic", out_dir: Path = DEFAULT_OUTPUT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "live":
        try:
            observations = collect_live()
        except Exception as exc:
            log.warning("live collection failed (%s); falling back to synthetic", exc)
            observations = build_synthetic_dataset()
            source = "synthetic-fallback"
    else:
        observations = build_synthetic_dataset()

    df = to_dataframe(observations)
    df.to_csv(out_dir / "observations.csv", index=False)

    cal = calibration_table(df)
    cal.to_csv(out_dir / "calibration_table.csv", index=False)

    plot_path = plot_reliability(df, out_dir / "reliability.png")

    sim = simulate_edge_strategy(df)
    sim.trades.to_csv(out_dir / "trades.csv", index=False)

    summary = {
        "source": source,
        "n_observations": int(len(df)),
        "brier": {
            "polymarket": brier_score(df["poly_prob"], df["outcome"]),
            "noaa":       brier_score(df["noaa_prob"], df["outcome"]),
        },
        "flagged_buckets": cal[cal["flagged"]].to_dict(orient="records"),
        "simulation": sim.summary,
        "bootstrap_95ci_mean_trade_roi": list(sim.bootstrap_ci),
        "reliability_plot": str(plot_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _print_human_summary(summary: dict) -> None:
    s = summary["simulation"]
    brier = summary["brier"]
    print(f"source                     : {summary['source']}")
    print(f"observations               : {summary['n_observations']}")
    print(f"Brier — Polymarket / NOAA  : {brier['polymarket']:.4f} / {brier['noaa']:.4f}")
    print(f"trades                     : {s['n_trades']}")
    print(f"win rate                   : {s['win_rate']:.2%}" if s['n_trades'] else "win rate                   : n/a")
    if s['n_trades']:
        print(f"net ROI (after 2% fees)    : {s['net_roi']:+.2%}")
        print(f"mean trade ROI             : {s['mean_trade_roi']:+.4f}")
        print(f"one-sided p-value (>0)     : {s['p_value_one_sided']:.4f}")
        print(f"bootstrap 95% CI of mean   : "
              f"[{summary['bootstrap_95ci_mean_trade_roi'][0]:+.4f}, "
              f"{summary['bootstrap_95ci_mean_trade_roi'][1]:+.4f}]")
        verdict = "YES" if s['significant_at_5pct'] else "NO"
        print(f"edge significant at p<0.05 : {verdict}")
    print(f"reliability plot           : {summary['reliability_plot']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["live", "synthetic"], default="synthetic")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run(source=args.source, out_dir=Path(args.out))
    _print_human_summary(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
