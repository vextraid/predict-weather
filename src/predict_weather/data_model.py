"""Shared data model for market observations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Iterable

import pandas as pd

CITY_COORDS: dict[str, tuple[float, float]] = {
    # Centroid lat/lon for the NOAA point-forecast endpoint.
    "NYC":     (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "Miami":   (25.7617, -80.1918),
    "Seattle": (47.6062, -122.3321),
    "Atlanta": (33.7490, -84.3880),
    "Dallas":  (32.7767, -96.7970),
}


@dataclass(frozen=True)
class MarketObservation:
    """One resolved Polymarket weather market aligned to a NOAA forecast.

    Probabilities are on [0, 1]. ``outcome`` is 1 if YES resolved, else 0.
    ``poly_prob`` is the J-1 mid-price (last trade ≤ 24h before resolution).
    ``noaa_prob`` is the NOAA-derived probability for the same event.
    """

    market_id: str
    title: str
    city: str
    target_date: date
    resolution_at: datetime
    poly_prob: float
    noaa_prob: float
    outcome: int

    def __post_init__(self) -> None:
        for name, p in (("poly_prob", self.poly_prob), ("noaa_prob", self.noaa_prob)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name}={p} outside [0, 1]")
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome={self.outcome} must be 0 or 1")
        if self.city not in CITY_COORDS:
            raise ValueError(f"unknown city {self.city!r}; add it to CITY_COORDS")


def to_dataframe(observations: Iterable[MarketObservation]) -> pd.DataFrame:
    """Materialize observations as a pandas DataFrame, one row per market."""
    rows = [asdict(o) for o in observations]
    if not rows:
        return pd.DataFrame(
            columns=[f.name for f in MarketObservation.__dataclass_fields__.values()]
        )
    df = pd.DataFrame(rows)
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["resolution_at"] = pd.to_datetime(df["resolution_at"])
    return df.sort_values("target_date").reset_index(drop=True)
