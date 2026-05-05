"""Open-Meteo forecast client (replaces NOAA for global city coverage).

Uses the free Open-Meteo API (https://api.open-meteo.com) — no API key needed.
Supports temperature-bucket and precipitation markets for any city in CITY_COORDS.

Market types supported:
  * "Highest temperature in <city> on <date>?" with buckets like "25°C", "76-77°F"
  * "Will it rain in <city> on <date>?"
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

from .data_model import CITY_COORDS, MarketObservation

log = logging.getLogger(__name__)

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
TEMP_CI_HALFWIDTH_C = 2.0  # ~90% CI half-width in °C for day-1 forecast


@dataclass
class OpenMeteoClient:
    timeout: float = 15.0
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.session.headers.setdefault("User-Agent", "predict-weather/0.2")

    def _fetch(self, city: str, target: date) -> dict:
        lat, lon = CITY_COORDS[city]
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
        }
        r = self.session.get(OPEN_METEO_BASE, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def probability_for(self, observation: MarketObservation) -> float:
        data = self._fetch(observation.city, observation.target_date)
        daily = data.get("daily", {})

        kind, threshold, unit = _classify_question(observation.title)

        if kind == "precip":
            vals = daily.get("precipitation_probability_max", [None])
            if not vals or vals[0] is None:
                return float("nan")
            return vals[0] / 100.0

        if kind in ("exact_temp", "temp_above", "temp_below", "temp_range"):
            max_vals = daily.get("temperature_2m_max", [None])
            min_vals = daily.get("temperature_2m_min", [None])
            if not max_vals or max_vals[0] is None:
                return float("nan")

            forecast_c = max_vals[0]
            # Convert threshold to Celsius if needed
            if unit == "F":
                threshold_c = (threshold - 32) * 5 / 9
                hi_c = ((threshold[1] - 32) * 5 / 9) if isinstance(threshold, tuple) else None
            else:
                threshold_c = threshold if not isinstance(threshold, tuple) else threshold[0]
                hi_c = threshold[1] if isinstance(threshold, tuple) else None

            sigma = TEMP_CI_HALFWIDTH_C / 1.645

            if kind == "exact_temp":
                # P(forecast within ±0.5°C of threshold)
                z_lo = ((threshold_c - 0.5) - forecast_c) / sigma
                z_hi = ((threshold_c + 0.5) - forecast_c) / sigma
                return _phi(z_hi) - _phi(z_lo)

            if kind == "temp_range" and hi_c is not None:
                z_lo = (threshold_c - forecast_c) / sigma
                z_hi = (hi_c - forecast_c) / sigma
                return _phi(z_hi) - _phi(z_lo)

            if kind == "temp_above":
                z = (threshold_c - forecast_c) / sigma
                return 1.0 - _phi(z)

            if kind == "temp_below":
                z = (threshold_c - forecast_c) / sigma
                return _phi(z)

        log.warning("unrecognised market title: %r", observation.title)
        return float("nan")


# Alias for backward compatibility with pipeline.py
NOAAClient = OpenMeteoClient


def _phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


# Regex patterns for title parsing
_RE_RAIN = re.compile(r"\b(rain|precip|snow|shower)\b", re.IGNORECASE)

# "25°C" or "25°F" — exact bucket
_RE_EXACT = re.compile(r"^(\d+)\s*°([CF])\s*$")

# "76-77°F" or "24-25°C" — range bucket
_RE_RANGE = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*°([CF])")

# "75°F or below", "80°F or higher"
_RE_THRESHOLD = re.compile(
    r"(\d+)\s*°([CF])\s+(or\s+)?(above|below|higher|lower|over|under)", re.IGNORECASE
)

# groupItemTitle style: "23°C", "76-77°F", "75°F or below"
def _classify_question(title: str) -> tuple[str, object, str]:
    if _RE_RAIN.search(title):
        return "precip", None, ""

    m = _RE_THRESHOLD.search(title)
    if m:
        val, unit, _, direction = float(m.group(1)), m.group(2), m.group(3), m.group(4).lower()
        if direction in ("above", "higher", "over"):
            return "temp_above", val, unit
        return "temp_below", val, unit

    m = _RE_RANGE.search(title)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
        return "temp_range", (lo, hi), unit

    m = _RE_EXACT.search(title)
    if m:
        return "exact_temp", float(m.group(1)), m.group(2)

    return "unknown", None, ""
