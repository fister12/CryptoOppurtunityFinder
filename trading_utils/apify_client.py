"""OHLCV data fetching — Apify with Binance public API fallback.

The Apify actor ``apify/crypto-ohlcv-scraper`` may or may not exist on the
marketplace. When Apify fails (bad actor, no token, network error), we fall
back to the free Binance public klines API which requires no authentication.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance public API (free, no auth)
# ---------------------------------------------------------------------------
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

ASSET_TO_BINANCE_SYMBOL = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
}

TIMEFRAME_TO_BINANCE = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


async def fetch_binance_ohlcv(
    asset: str,
    limit: int = 1000,
    timeframe: str = "1m",
    request_timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance's public klines endpoint.

    Requires no authentication. Returns a normalised DataFrame with
    columns: open, high, low, close, volume, timestamps.
    """
    symbol = ASSET_TO_BINANCE_SYMBOL.get(asset.upper())
    if symbol is None:
        raise ValueError(f"Unknown asset '{asset}' — supported: {list(ASSET_TO_BINANCE_SYMBOL)}")

    interval = TIMEFRAME_TO_BINANCE.get(timeframe, timeframe)
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}

    async with httpx.AsyncClient(timeout=request_timeout) as client:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.get(BINANCE_KLINES_URL, params=params)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(2 ** attempt)
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Binance klines request failed")

    # Binance klines format: [open_time, open, high, low, close, volume, ...]
    rows = []
    for candle in data:
        rows.append({
            "timestamps": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Binance returned no candles")

    frame = frame.sort_values("timestamps").tail(limit).reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Apify client (primary, when APIFY_API_TOKEN is set)
# ---------------------------------------------------------------------------
class ApifyOHLCVClient:
    """Fetch OHLCV data via Apify actors.

    Falls back to Binance public API if:
    - No APIFY_API_TOKEN is configured
    - The actor run fails
    - The dataset returns no data
    """

    def __init__(self, settings) -> None:
        self.settings = settings

    async def fetch_ohlcv(
        self,
        asset: str,
        limit: int = 1000,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Try Apify first, fall back to Binance on any failure."""
        if self.settings.apify_api_token:
            try:
                return await self._fetch_via_apify(asset, limit, timeframe)
            except Exception as exc:
                logger.warning(
                    "Apify fetch failed for %s (actor=%s): %s — falling back to Binance",
                    asset,
                    self.settings.apify_actor_id,
                    exc,
                )

        logger.info("Fetching %s OHLCV via Binance public API (%s, limit=%d)", asset, timeframe, limit)
        return await fetch_binance_ohlcv(
            asset=asset,
            limit=limit,
            timeframe=timeframe,
            request_timeout=self.settings.request_timeout_seconds,
        )

    async def _fetch_via_apify(
        self,
        asset: str,
        limit: int,
        timeframe: str,
    ) -> pd.DataFrame:
        from apify_client import ApifyClientAsync

        client = ApifyClientAsync(self.settings.apify_api_token)
        actor = client.actor(self.settings.apify_actor_id)
        run_input = {
            "asset": asset,
            "symbol": asset,
            "limit": limit,
            "timeframe": timeframe,
        }

        run = await self._retry_async(actor.call, run_input=run_input)
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "defaultDatasetId", None)
        if not dataset_id:
            raise RuntimeError("Apify run did not return a dataset id")

        dataset = client.dataset(dataset_id)
        items_response = await self._retry_async(dataset.list_items, limit=limit)
        items = self._extract_items(items_response)
        frame = self._items_to_frame(items)
        if frame.empty:
            raise RuntimeError("Apify dataset returned no OHLCV candles")
        return frame

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _retry_async(self, func, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        retries = max(1, int(getattr(self.settings, "max_retries", 3)))
        for attempt in range(retries):
            try:
                value = func(**kwargs)
                return await self._maybe_await(value)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= retries:
                    break
                await asyncio.sleep(2 ** attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Apify request failed without a captured exception")

    def _extract_items(self, response: Any) -> list[dict[str, Any]]:
        if isinstance(response, dict):
            items = response.get("items") or response.get("data") or []
            return [item for item in items if isinstance(item, dict)]
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        return []

    def _items_to_frame(self, items: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(items)
        if frame.empty:
            return frame

        column_map = {
            "timestamp": "timestamps",
            "time": "timestamps",
            "datetime": "timestamps",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        for source_column, target_column in column_map.items():
            if source_column in frame.columns and target_column not in frame.columns:
                frame[target_column] = frame[source_column]

        required = ["open", "high", "low", "close", "volume", "timestamps"]
        for column in required:
            if column not in frame.columns:
                raise RuntimeError(f"Apify response is missing required column: {column}")

        frame = frame[required].copy()
        frame["timestamps"] = pd.to_datetime(frame["timestamps"], utc=True, errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna().sort_values("timestamps").tail(1000).reset_index(drop=True)
        return frame
