"""Polymarket gamma-api client for resolved weather markets.

Endpoints used (public, no auth required):
- GET https://gamma-api.polymarket.com/markets         -> market metadata
- GET https://clob.polymarket.com/prices-history       -> minute-level mid prices

Notes
-----
Polymarket weather markets are sporadic; we filter by:
  * the ``weather`` tag/category, AND
  * a substring match on city names in the question text.

The J-1 reference price is the mid-price of the last trade strictly before
``resolution_at - 24h``. If no trade exists in that window we skip the market.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator

import requests

from .data_model import CITY_COORDS, MarketObservation

log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
PAGE_LIMIT = 100

_CITY_PATTERNS = {
    city: re.compile(rf"\b{re.escape(city)}\b", re.IGNORECASE)
    for city in CITY_COORDS
}
# Aliases that show up in market titles.
_CITY_PATTERNS["NYC"] = re.compile(r"\b(NYC|New York( City)?)\b", re.IGNORECASE)


def _detect_city(title: str) -> str | None:
    for city, pattern in _CITY_PATTERNS.items():
        if pattern.search(title):
            return city
    return None


@dataclass
class PolymarketClient:
    """Thin wrapper around Polymarket's public gamma + CLOB APIs."""

    timeout: float = 15.0
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.session.headers.setdefault("User-Agent", "predict-weather/0.1")

    # ------------------------------------------------------------------ #
    # market discovery
    # ------------------------------------------------------------------ #
    def iter_resolved_weather_markets(
        self,
        since: datetime,
        until: datetime | None = None,
    ) -> Iterator[dict]:
        """Yield raw market dicts resolved between ``since`` and ``until``."""
        until = until or datetime.now(timezone.utc)
        offset = 0
        while True:
            params = {
                "closed": "true",
                "limit": PAGE_LIMIT,
                "offset": offset,
                "tag_slug": "weather",
                "end_date_min": since.isoformat(),
                "end_date_max": until.isoformat(),
            }
            r = self.session.get(
                f"{GAMMA_BASE}/markets", params=params, timeout=self.timeout
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                return
            yield from batch
            if len(batch) < PAGE_LIMIT:
                return
            offset += PAGE_LIMIT

    # ------------------------------------------------------------------ #
    # prices
    # ------------------------------------------------------------------ #
    def price_at(self, token_id: str, ts: datetime) -> float | None:
        """Return the YES mid-price at ``ts``, or None if no trade is found."""
        params = {
            "market": token_id,
            "startTs": int((ts - timedelta(hours=6)).timestamp()),
            "endTs": int(ts.timestamp()),
            "fidelity": 60,  # 1-minute resolution
        }
        r = self.session.get(
            f"{CLOB_BASE}/prices-history", params=params, timeout=self.timeout
        )
        r.raise_for_status()
        history = r.json().get("history", [])
        if not history:
            return None
        # last point at or before ts
        history.sort(key=lambda x: x["t"])
        last = history[-1]
        return float(last["p"])

    # ------------------------------------------------------------------ #
    # high-level
    # ------------------------------------------------------------------ #
    def collect_observations(
        self, since: datetime, until: datetime | None = None
    ) -> list[MarketObservation]:
        """Collect resolved weather markets as MarketObservation rows.

        Returns markets whose title matches one of the target cities and that
        resolved cleanly to YES or NO with a price tick at J-1.
        """
        observations: list[MarketObservation] = []
        for raw in self.iter_resolved_weather_markets(since, until):
            try:
                obs = self._build_observation(raw)
            except _SkipMarket as exc:
                log.debug("skipping market %s: %s", raw.get("id"), exc)
                continue
            if obs is not None:
                observations.append(obs)
        return observations

    def _build_observation(self, raw: dict) -> MarketObservation | None:
        title = raw.get("question") or raw.get("title") or ""
        city = _detect_city(title)
        if city is None:
            raise _SkipMarket("no target city in title")

        end_iso = raw.get("endDate") or raw.get("end_date")
        if end_iso is None:
            raise _SkipMarket("missing endDate")
        resolution_at = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        outcomes = raw.get("outcomePrices") or raw.get("outcome_prices") or []
        if isinstance(outcomes, str):
            # gamma sometimes returns a JSON string
            import json
            outcomes = json.loads(outcomes)
        if len(outcomes) < 2:
            raise _SkipMarket("non-binary outcomes")
        yes_settle = float(outcomes[0])
        outcome = 1 if yes_settle > 0.5 else 0

        token_id = raw.get("clobTokenIds", [None])[0] or raw.get("conditionId")
        if token_id is None:
            raise _SkipMarket("missing token id")

        poly_prob = self.price_at(token_id, resolution_at - timedelta(hours=24))
        if poly_prob is None:
            raise _SkipMarket("no J-1 trade")

        return MarketObservation(
            market_id=str(raw.get("id") or raw.get("conditionId")),
            title=title,
            city=city,
            target_date=resolution_at.date(),
            resolution_at=resolution_at,
            poly_prob=poly_prob,
            noaa_prob=float("nan"),  # filled in by NOAA client
            outcome=outcome,
        )


class _SkipMarket(Exception):
    """Internal: signal that a raw market should be skipped."""
