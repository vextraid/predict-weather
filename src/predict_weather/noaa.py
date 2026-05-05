"""NOAA forecast client.

Uses the public NWS API (https://api.weather.gov) for *current* forecasts.
For a true historical backtest you must instead pull the NDFD archive from
NCEI (https://www.ncei.noaa.gov/products/weather-climate-models/ndfd) — that
archive contains the gridded forecast as it stood at any past time T. The
``HistoricalForecastSource`` protocol below makes that pluggable.

The mapping from a NOAA forecast to a Polymarket-style binary probability
depends on the question. We support:

  * ``probabilityOfPrecipitation`` for "Will it rain in <city> on <date>?"
  * a temperature-threshold heuristic that converts forecast min/max + a
    known 90-percent-CI band into a normal-tail probability for questions
    of the form "Will <city> exceed/below T°F?".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

import math
import requests

from .data_model import CITY_COORDS, MarketObservation

log = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
# Empirical 24h NWS day-1 forecast spread (≈90% CI half-width in °F).
TEMP_CI_HALFWIDTH_F = 4.0


class HistoricalForecastSource(Protocol):
    """Plug a real NDFD-archive backend in production."""

    def forecast(self, city: str, target: date, issued_at: datetime) -> dict:
        ...


@dataclass
class NOAAClient:
    """NWS forecast client. Production: replace ``source`` with NDFD archive."""

    timeout: float = 15.0
    session: requests.Session | None = None
    source: HistoricalForecastSource | None = None

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "predict-weather/0.1 (research; contact@example.com)"
        )
        self.session.headers.setdefault("Accept", "application/geo+json")

    # ------------------------------------------------------------------ #
    # raw NWS endpoints
    # ------------------------------------------------------------------ #
    def _gridpoint(self, city: str) -> tuple[str, int, int]:
        lat, lon = CITY_COORDS[city]
        r = self.session.get(
            f"{NWS_BASE}/points/{lat},{lon}", timeout=self.timeout
        )
        r.raise_for_status()
        props = r.json()["properties"]
        return props["gridId"], props["gridX"], props["gridY"]

    def _gridpoint_forecast(self, city: str) -> dict:
        wfo, gx, gy = self._gridpoint(city)
        r = self.session.get(
            f"{NWS_BASE}/gridpoints/{wfo}/{gx},{gy}", timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["properties"]

    # ------------------------------------------------------------------ #
    # market-question -> probability
    # ------------------------------------------------------------------ #
    def probability_for(self, observation: MarketObservation) -> float:
        """Map an observation's title to a NOAA probability."""
        if self.source is not None:
            grid = self.source.forecast(
                observation.city,
                observation.target_date,
                observation.resolution_at - timedelta(hours=24),
            )
        else:
            grid = self._gridpoint_forecast(observation.city)

        kind, threshold = _classify_question(observation.title)
        if kind == "precip":
            return _max_pop_for_day(grid, observation.target_date)
        if kind == "max_above":
            return _temperature_tail_probability(
                grid, observation.target_date, threshold, "above", "max"
            )
        if kind == "min_below":
            return _temperature_tail_probability(
                grid, observation.target_date, threshold, "below", "min"
            )
        log.warning("unrecognised market title: %r — defaulting to NaN", observation.title)
        return float("nan")


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
_RE_RAIN = re.compile(r"\b(rain|precip|snow|shower)\b", re.IGNORECASE)
_RE_TEMP = re.compile(
    r"(high|max|low|min)\s.*?(above|below|over|under|exceed|>=|<=|>|<)\s*(-?\d{1,3})",
    re.IGNORECASE,
)


def _classify_question(title: str) -> tuple[str, float | None]:
    if _RE_RAIN.search(title):
        return "precip", None
    m = _RE_TEMP.search(title)
    if m:
        bound = m.group(1).lower()
        direction = m.group(2).lower()
        threshold = float(m.group(3))
        if bound in ("high", "max") and direction in ("above", "over", "exceed", ">", ">="):
            return "max_above", threshold
        if bound in ("low", "min") and direction in ("below", "under", "<", "<="):
            return "min_below", threshold
    return "unknown", None


def _series(grid: dict, key: str) -> list[dict]:
    return grid.get(key, {}).get("values", [])


def _values_on(values: list[dict], target: date) -> list[float]:
    out: list[float] = []
    for entry in values:
        ts = entry.get("validTime", "").split("/")[0]
        if not ts:
            continue
        d = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).date()
        if d == target and entry.get("value") is not None:
            out.append(float(entry["value"]))
    return out


def _max_pop_for_day(grid: dict, target: date) -> float:
    pops = _values_on(_series(grid, "probabilityOfPrecipitation"), target)
    if not pops:
        return float("nan")
    return max(pops) / 100.0


def _temperature_tail_probability(
    grid: dict,
    target: date,
    threshold_f: float,
    direction: str,
    bound: str,
) -> float:
    series_key = "maxTemperature" if bound == "max" else "minTemperature"
    celsius_vals = _values_on(_series(grid, series_key), target)
    if not celsius_vals:
        return float("nan")
    point_estimate_f = celsius_vals[0] * 9 / 5 + 32  # NWS gridpoints are °C
    sigma = TEMP_CI_HALFWIDTH_F / 1.645  # 90 % CI half-width -> 1σ
    z = (threshold_f - point_estimate_f) / sigma
    # P(T > threshold)
    p_above = 0.5 * math.erfc(z / math.sqrt(2))
    return p_above if direction == "above" else 1.0 - p_above
