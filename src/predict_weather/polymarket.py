"""Polymarket gamma-api client for weather markets.

Fetches ACTIVE (open) weather markets and extracts:
- City from title ("Highest temperature in Dallas on May 6?")
- Temperature bucket from groupItemTitle ("25°C", "76-77°F", "75°F or below")
- J-1 mid-price from CLOB
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import requests

from .data_model import CITY_COORDS, MarketObservation

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
PAGE_LIMIT = 100

# Match city name in title
_CITY_RE = {
    city: re.compile(rf"\b{re.escape(city)}\b", re.IGNORECASE)
    for city in CITY_COORDS
}
_CITY_RE["NYC"] = re.compile(r"\b(NYC|New York( City)?)\b", re.IGNORECASE)
_CITY_RE["Hong Kong"] = re.compile(r"\bHong Kong\b", re.IGNORECASE)

# Match date in title: "on May 6", "on May 6?"
_DATE_RE = re.compile(
    r"\bon\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})",
    re.IGNORECASE,
)
_MONTH = {m: i for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 1
)}


def _detect_city(title: str) -> str | None:
    for city, pat in _CITY_RE.items():
        if pat.search(title):
            return city
    return None


def _detect_date(title: str, reference: datetime) -> datetime | None:
    m = _DATE_RE.search(title)
    if not m:
        return None
    month = _MONTH[m.group(1).lower()]
    day   = int(m.group(2))
    year  = reference.year
    # Roll over to next year if the date is in the past
    try:
        d = datetime(year, month, day, 12, 0, tzinfo=timezone.utc)
        if d < reference - timedelta(days=1):
            d = datetime(year + 1, month, day, 12, 0, tzinfo=timezone.utc)
        return d
    except ValueError:
        return None


@dataclass
class PolymarketClient:
    timeout: float = 15.0
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.session.headers.setdefault("User-Agent", "predict-weather/0.2")

    def iter_weather_markets(self) -> Iterator[dict]:
        """Yield active weather markets from gamma API."""
        offset = 0
        while True:
            params = {
                "active": "true",
                "closed": "false",
                "limit": PAGE_LIMIT,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false",
            }
            r = self.session.get(
                f"{GAMMA_BASE}/markets", params=params, timeout=self.timeout
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                return
            for m in batch:
                title = m.get("question") or m.get("title") or ""
                if "temperature" in title.lower() or "rain" in title.lower():
                    yield m
            offset += PAGE_LIMIT
            if offset >= 600:
                return

    def price_at(self, token_id: str, ts: datetime) -> float | None:
        params = {
            "market": token_id,
            "startTs": int((ts - timedelta(hours=6)).timestamp()),
            "endTs": int(ts.timestamp()),
            "fidelity": 60,
        }
        r = self.session.get(
            f"{CLOB_BASE}/prices-history", params=params, timeout=self.timeout
        )
        r.raise_for_status()
        history = r.json().get("history", [])
        if not history:
            return None
        history.sort(key=lambda x: x["t"])
        return float(history[-1]["p"])

    def collect_observations(self, since=None, until=None) -> list[MarketObservation]:
        now = datetime.now(timezone.utc)
        observations: list[MarketObservation] = []
        for raw in self.iter_weather_markets():
            try:
                obs = self._build_observation(raw, now)
            except _Skip as e:
                log.debug("skip %s: %s", raw.get("id"), e)
                continue
            if obs is not None:
                observations.append(obs)
        return observations

    def _build_observation(self, raw: dict, now: datetime) -> MarketObservation | None:
        # Use groupItemTitle for bucket label, question for city/date
        question     = raw.get("question") or raw.get("title") or ""
        bucket_label = raw.get("groupItemTitle") or ""

        city = _detect_city(question)
        if city is None:
            raise _Skip("no target city")

        end_iso = raw.get("endDate") or raw.get("end_date")
        if not end_iso:
            raise _Skip("no endDate")
        resolution_at = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        # Use bucket label for NOAA classification
        noaa_title = bucket_label if bucket_label else question

        # For active markets, outcome is unknown — use current best_ask as proxy
        best_ask = raw.get("bestAsk") or raw.get("lastTradePrice")
        if best_ask is None:
            raise _Skip("no price")
        poly_prob = float(best_ask)
        if not 0.0 < poly_prob < 1.0:
            raise _Skip(f"degenerate price {poly_prob}")

        return MarketObservation(
            market_id    = str(raw.get("id") or raw.get("conditionId")),
            title        = noaa_title,
            city         = city,
            target_date  = resolution_at.date(),
            resolution_at= resolution_at,
            poly_prob    = poly_prob,
            noaa_prob    = float("nan"),
            outcome      = 0,  # unknown for active markets
        )


class _Skip(Exception):
    pass
