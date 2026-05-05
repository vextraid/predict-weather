"""Reproducible synthetic dataset generator.

Used for offline testing of the pipeline when the live APIs are unreachable.
The generator deliberately injects a calibration bias so the analysis layer
has something to detect:

  * NOAA is treated as the (noisy) ground-truth probability.
  * Polymarket J-1 mid-prices are pulled toward 0.5 with extra noise — this
    mimics the empirical observation that retail prediction markets on
    low-volume questions are under-confident at the tails.

Real data: replace the call to ``build_synthetic_dataset`` with the real
PolymarketClient + NOAAClient pipeline (see ``pipeline.py``).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from .data_model import CITY_COORDS, MarketObservation

CITIES = list(CITY_COORDS.keys())

QUESTION_TEMPLATES = [
    ("rain", "Will it rain in {city} on {date}?"),
    ("max_above", "Will the high temperature in {city} on {date} be above {threshold}°F?"),
    ("min_below", "Will the low temperature in {city} on {date} be below {threshold}°F?"),
]


def build_synthetic_dataset(
    *,
    n: int = 240,
    start: date | None = None,
    end: date | None = None,
    bias_strength: float = 0.30,
    rng_seed: int = 42,
) -> list[MarketObservation]:
    """Generate ``n`` resolved market observations spanning ``[start, end]``.

    Parameters
    ----------
    bias_strength
        How strongly Polymarket pulls toward 0.5. 0 = perfectly calibrated,
        1 = always flat at 0.5.
    """
    rng = np.random.default_rng(rng_seed)
    end = end or date(2026, 5, 1)
    start = start or end - timedelta(days=185)
    span_days = (end - start).days

    observations: list[MarketObservation] = []
    for i in range(n):
        offset_days = int(rng.integers(0, span_days + 1))
        target = start + timedelta(days=offset_days)
        city = CITIES[int(rng.integers(0, len(CITIES)))]
        kind, template = QUESTION_TEMPLATES[int(rng.integers(0, len(QUESTION_TEMPLATES)))]

        # NOAA is "true" probability with mild dispersion across [0.05, 0.95]
        noaa_prob = float(np.clip(rng.beta(2, 2), 0.05, 0.95))

        # Polymarket = NOAA shrunk toward 0.5 + idiosyncratic noise
        poly_prob = float(np.clip(
            0.5 + (noaa_prob - 0.5) * (1 - bias_strength) + rng.normal(0, 0.05),
            0.02, 0.98,
        ))

        outcome = int(rng.random() < noaa_prob)

        if kind == "rain":
            title = template.format(city=city, date=target.isoformat())
        else:
            threshold = int(rng.integers(40, 95))
            title = template.format(city=city, date=target.isoformat(), threshold=threshold)

        resolution_at = datetime.combine(
            target + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
        )
        observations.append(MarketObservation(
            market_id=f"synthetic-{i:04d}",
            title=title,
            city=city,
            target_date=target,
            resolution_at=resolution_at,
            poly_prob=poly_prob,
            noaa_prob=noaa_prob,
            outcome=outcome,
        ))
    return observations
