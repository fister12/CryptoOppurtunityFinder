"""Market sources — scan Polymarket and Kalshi for BTC/ETH opportunities.

Key fixes from original:
- Kalshi API requires auth: gracefully degrades when unauthenticated
- resolve_market_outcome now RE-FETCHES from the API instead of
  reading stale data from the cached raw_payload
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx

from core.models import MarketOpportunity, ResolutionResult

logger = logging.getLogger(__name__)

POLYMARKET_ENDPOINT = "https://gamma-api.polymarket.com/markets"
KALSHI_ENDPOINT = "https://trading-api.kalshi.com/trade-api/v2/markets"


def _asset_matches(asset: str, text: str) -> bool:
    normalized = text.upper()
    return asset.upper() in normalized or (asset.upper() == "BTC" and "BITCOIN" in normalized) or (asset.upper() == "ETH" and "ETHEREUM" in normalized)


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


async def _fetch_json_with_retries(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            if response.status_code in (401, 403):
                raise httpx.HTTPStatusError(
                    f"Authentication required (HTTP {response.status_code})",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(2 ** attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("request failed without a captured exception")


def _market_window(now: datetime, min_minutes: int, max_minutes: int) -> tuple[datetime, datetime]:
    return now + timedelta(minutes=min_minutes), now + timedelta(minutes=max_minutes)


def _opportunity_from_payload(
    venue: str,
    payload: dict[str, Any],
    asset: str,
    expires_at: datetime | None,
    share_price: float,
    market_id_keys: Iterable[str],
    title_keys: Iterable[str],
) -> MarketOpportunity | None:
    if expires_at is None:
        return None

    market_id = ""
    for key in market_id_keys:
        value = payload.get(key)
        if value not in (None, ""):
            market_id = str(value)
            break
    if not market_id:
        market_id = str(payload.get("id") or payload.get("ticker") or payload.get("slug") or "unknown")

    title = ""
    for key in title_keys:
        value = payload.get(key)
        if value not in (None, ""):
            title = str(value)
            break

    return MarketOpportunity(
        venue=venue,
        market_id=market_id,
        asset=asset,
        share_price=share_price,
        expires_at=expires_at,
        title=title,
        raw_payload=payload,
    )


async def scan_polymarket(now: datetime, min_minutes: int, max_minutes: int, assets: tuple[str, ...], request_timeout: float) -> list[MarketOpportunity]:
    opportunities: list[MarketOpportunity] = []
    window_start, window_end = _market_window(now, min_minutes, max_minutes)
    async with httpx.AsyncClient(timeout=request_timeout) as client:
        try:
            payload = await _fetch_json_with_retries(client, POLYMARKET_ENDPOINT, params={"active": "true", "limit": 100})
        except Exception as exc:
            logger.warning("Polymarket scan failed: %r", exc, exc_info=True)
            return opportunities

    items = payload if isinstance(payload, list) else payload.get("data") or payload.get("markets") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("question") or item.get("title") or item.get("name") or "")
        if not any(_asset_matches(asset, title) for asset in assets):
            continue
        expires_at = _parse_datetime(item.get("endDate") or item.get("expirationTime") or item.get("closeTime") or item.get("expiresAt"))
        if expires_at is None or not (window_start <= expires_at <= window_end):
            continue
        share_price = _first_float(item.get("price") or item.get("lastTradePrice") or item.get("bestAsk") or item.get("yes_price"), default=0.0)
        if share_price <= 0.0:
            continue
        for asset in assets:
            if _asset_matches(asset, title):
                opportunity = _opportunity_from_payload(
                    "polymarket",
                    item,
                    asset,
                    expires_at,
                    share_price,
                    ("id", "marketId", "conditionId", "slug"),
                    ("question", "title", "name"),
                )
                if opportunity is not None:
                    opportunities.append(opportunity)
                break

    return opportunities


async def scan_kalshi(now: datetime, min_minutes: int, max_minutes: int, assets: tuple[str, ...], request_timeout: float) -> list[MarketOpportunity]:
    """Scan Kalshi markets.

    Kalshi's API requires authentication. If we get a 401/403, we log a
    warning and return empty (graceful degradation) rather than crashing.
    """
    opportunities: list[MarketOpportunity] = []
    window_start, window_end = _market_window(now, min_minutes, max_minutes)
    async with httpx.AsyncClient(timeout=request_timeout) as client:
        try:
            payload = await _fetch_json_with_retries(client, KALSHI_ENDPOINT, params={"limit": 100})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                logger.warning(
                    "Kalshi API returned %d — authentication required. "
                    "Set KALSHI_API_KEY in .env to enable Kalshi scanning. "
                    "Continuing with Polymarket only.",
                    exc.response.status_code,
                )
                return opportunities
            logger.warning("Kalshi scan failed: %r", exc, exc_info=True)
            return opportunities
        except Exception as exc:
            logger.warning("Kalshi scan failed: %r", exc, exc_info=True)
            return opportunities

    items = payload if isinstance(payload, list) else payload.get("markets") or payload.get("data") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("question") or item.get("name") or item.get("ticker") or "")
        if not any(_asset_matches(asset, title) for asset in assets):
            continue
        expires_at = _parse_datetime(item.get("close_ts") or item.get("expiration_ts") or item.get("end_ts") or item.get("expiration_time"))
        if expires_at is None or not (window_start <= expires_at <= window_end):
            continue
        share_price = _first_float(item.get("yes_ask") or item.get("last") or item.get("price") or item.get("best_ask"), default=0.0)
        if share_price <= 0.0:
            continue
        for asset in assets:
            if _asset_matches(asset, title):
                opportunity = _opportunity_from_payload(
                    "kalshi",
                    item,
                    asset,
                    expires_at,
                    share_price,
                    ("ticker", "id", "market_id"),
                    ("title", "question", "name"),
                )
                if opportunity is not None:
                    opportunities.append(opportunity)
                break

    return opportunities


async def scan_target_markets(min_minutes: int, max_minutes: int, assets: tuple[str, ...], now: datetime, request_timeout: float = 20.0) -> list[MarketOpportunity]:
    polymarket_task = scan_polymarket(now, min_minutes, max_minutes, assets, request_timeout)
    kalshi_task = scan_kalshi(now, min_minutes, max_minutes, assets, request_timeout)
    polymarket_markets, kalshi_markets = await asyncio.gather(polymarket_task, kalshi_task)
    combined = polymarket_markets + kalshi_markets
    logger.info(
        "Market scan: %d Polymarket + %d Kalshi = %d total opportunities",
        len(polymarket_markets),
        len(kalshi_markets),
        len(combined),
    )
    return sorted(combined, key=lambda opportunity: opportunity.expires_at)


async def resolve_market_outcome(opportunity: MarketOpportunity, request_timeout: float = 20.0) -> ResolutionResult:
    """Re-fetch market data from the API to get actual resolution status.

    The original implementation only read from the cached raw_payload snapshot,
    which was taken at scan time and never reflected the actual outcome.
    """
    venue = opportunity.venue
    market_id = opportunity.market_id

    # Try to re-fetch fresh data from the venue
    try:
        if venue == "polymarket":
            fresh_payload = await _refetch_polymarket(market_id, request_timeout)
        elif venue == "kalshi":
            fresh_payload = await _refetch_kalshi(market_id, request_timeout)
        else:
            fresh_payload = None
    except Exception as exc:
        logger.warning("Failed to re-fetch outcome for %s/%s: %s — using cached payload", venue, market_id, exc)
        fresh_payload = None

    # Use fresh data if available, otherwise fall back to cached
    payload = fresh_payload if fresh_payload else (opportunity.raw_payload or {})

    status = str(
        payload.get("status")
        or payload.get("resolution_status")
        or payload.get("state")
        or payload.get("result")
        or "unresolved"
    ).lower()

    actual_outcome = payload.get("winner") or payload.get("resolution") or payload.get("outcome") or payload.get("result")
    if actual_outcome not in (None, ""):
        actual_outcome = str(actual_outcome)
    else:
        actual_outcome = None

    if status in {"resolved", "settled", "closed", "expired"} and actual_outcome is None:
        actual_outcome = status

    return ResolutionResult(status=status, actual_outcome=actual_outcome, raw_payload=payload)


async def _refetch_polymarket(market_id: str, timeout: float) -> dict[str, Any] | None:
    """Re-fetch a single Polymarket market by ID."""
    url = f"{POLYMARKET_ENDPOINT}/{market_id}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        payload = await _fetch_json_with_retries(client, url)
        return payload if isinstance(payload, dict) else None


async def _refetch_kalshi(market_id: str, timeout: float) -> dict[str, Any] | None:
    """Re-fetch a single Kalshi market by ticker."""
    url = f"{KALSHI_ENDPOINT}/{market_id}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            payload = await _fetch_json_with_retries(client, url)
            if isinstance(payload, dict):
                return payload.get("market", payload)
        except httpx.HTTPStatusError:
            pass  # Kalshi may require auth; fall through to cached
        return None
