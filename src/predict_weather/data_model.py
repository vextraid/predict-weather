"""Shared data model for market observations."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Iterable

import pandas as pd

CITY_COORDS: dict[str, tuple[float, float]] = {
    "NYC":        (40.7128,  -74.0060),
    "Chicago":    (41.8781,  -87.6298),
    "Miami":      (25.7617,  -80.1918),
    "Seattle":    (47.6062, -122.3321),
    "Atlanta":    (33.7490,  -84.3880),
    "Dallas":     (32.7767,  -96.7970),
    "London":     (51.5074,   -0.1278),
    "Tokyo":      (35.6762,  139.6503),
    "Shanghai":   (31.2304,  121.4737),
    "Seoul":      (37.5665,  126.9780),
    "Wellington": (-41.2866, 174.7756),
    "Paris":      (48.8566,    2.3522),
    "Sydney":     (-33.8688, 151.2093),
    "Singapore":  (1.3521,   103.8198),
    "Dubai":      (25.2048,   55.2708),
    "Hong Kong":  (22.3193,  114.1694),
    "Berlin":     (52.5200,   13.4050),
    "Madrid":     (40.4168,   -3.7038),
    "Bangkok":    (13.7563,  100.5018),
    "Chongqing":  (29.4316,  106.9123),
}


@dataclass(frozen=True)
class MarketObservation:
    """One resolved Polymarket weather market aligned to a forecast.

    Probabilities are on [0, 1]. ``outcome`` is 1 if YES resolved, else 0.
    ``poly_prob`` is the J-1 mid-price (last trade <= 24h before resolution).
    ``noaa_prob`` is the forecast-derived probability for the same event.
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
        import math
        for name, p in (("poly_prob", self.poly_prob), ("noaa_prob", self.noaa_prob)):
            if math.isnan(p):
                continue
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
