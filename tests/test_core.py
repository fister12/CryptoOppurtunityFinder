"""Unit tests for the crypto trading agent system."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.models import MarketOpportunity, PredictionOutcome, RiskDecision
from core.risk import calculate_kelly_fraction, calculate_stake
from models.kronos import normalize_ohlcv_frame, frame_to_ohlcv_array, infer_direction, summarize_frame


# ---------------------------------------------------------------------------
# Kelly Criterion Tests
# ---------------------------------------------------------------------------
class TestKellyCriterion:
    def test_kelly_fraction_basic(self):
        """60% confidence at 50% share price → positive Kelly fraction."""
        fraction = calculate_kelly_fraction(confidence=0.6, share_price=0.5)
        assert fraction > 0.0
        assert fraction < 1.0

    def test_kelly_fraction_no_edge(self):
        """50% confidence at 50% odds → zero (no edge)."""
        fraction = calculate_kelly_fraction(confidence=0.5, share_price=0.5)
        assert fraction == 0.0

    def test_kelly_fraction_negative_edge(self):
        """30% confidence → no bet (negative edge)."""
        fraction = calculate_kelly_fraction(confidence=0.3, share_price=0.5)
        assert fraction == 0.0

    def test_kelly_fraction_high_confidence(self):
        """90% confidence → large fraction."""
        fraction = calculate_kelly_fraction(confidence=0.9, share_price=0.5)
        assert fraction > 0.5

    def test_kelly_fraction_extreme_price(self):
        """Share price at boundary → zero."""
        assert calculate_kelly_fraction(confidence=0.7, share_price=0.0) == 0.0
        assert calculate_kelly_fraction(confidence=0.7, share_price=1.0) == 0.0

    def test_calculate_stake(self):
        """Stake calculation with fractional Kelly."""
        decision = calculate_stake(
            bankroll_usd=1000.0,
            confidence=0.7,
            share_price=0.5,
            fractional_kelly=0.5,
        )
        assert isinstance(decision, RiskDecision)
        assert decision.stake_usd > 0.0
        assert decision.stake_usd <= 1000.0
        assert decision.fractional_kelly_fraction < decision.raw_kelly_fraction

    def test_calculate_stake_no_bet(self):
        """Low confidence → zero stake."""
        decision = calculate_stake(
            bankroll_usd=1000.0,
            confidence=0.3,
            share_price=0.5,
        )
        assert decision.stake_usd == 0.0


# ---------------------------------------------------------------------------
# OHLCV Normalisation Tests
# ---------------------------------------------------------------------------
def _make_ohlcv_frame(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    base_price = 50000.0
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    closes = base_price + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({
        "timestamps": timestamps,
        "open": closes - np.random.rand(n) * 5,
        "high": closes + np.random.rand(n) * 10,
        "low": closes - np.random.rand(n) * 10,
        "close": closes,
        "volume": np.random.rand(n) * 1000,
    })


class TestOHLCVNormalization:
    def test_normalize_valid_frame(self):
        frame = _make_ohlcv_frame()
        normalized = normalize_ohlcv_frame(frame)
        assert len(normalized) == 100
        assert set(normalized.columns) == {"open", "high", "low", "close", "volume", "timestamps"}

    def test_normalize_missing_columns(self):
        frame = pd.DataFrame({"open": [1, 2], "close": [3, 4]})
        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_ohlcv_frame(frame)

    def test_normalize_with_nans(self):
        frame = _make_ohlcv_frame(50)
        frame.loc[10, "close"] = None
        frame.loc[20, "open"] = np.nan
        normalized = normalize_ohlcv_frame(frame)
        assert len(normalized) == 48  # 2 rows dropped

    def test_frame_to_array(self):
        frame = _make_ohlcv_frame(20)
        arr = frame_to_ohlcv_array(frame)
        assert arr.shape == (20, 5)
        assert arr.dtype == np.float32

    def test_summarize_frame(self):
        frame = _make_ohlcv_frame()
        summary = summarize_frame(frame)
        assert "last_close" in summary
        assert "volatility" in summary
        assert summary["row_count"] == 100

    def test_infer_direction(self):
        assert infer_direction(100.0, 99.0) == "Up"
        assert infer_direction(99.0, 100.0) == "Down"
        assert infer_direction(100.0, 100.0) == "Up"


# ---------------------------------------------------------------------------
# Market Opportunity Parsing Tests
# ---------------------------------------------------------------------------
class TestMarketSources:
    def test_asset_matches(self):
        from core.market_sources import _asset_matches
        assert _asset_matches("BTC", "Will Bitcoin exceed $100k?")
        assert _asset_matches("ETH", "Ethereum price prediction")
        assert not _asset_matches("BTC", "Will gold price rise?")
        assert _asset_matches("BTC", "BTC/USDT 5-min prediction")

    def test_parse_datetime(self):
        from core.market_sources import _parse_datetime
        dt = _parse_datetime("2025-06-29T12:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

        assert _parse_datetime(None) is None
        assert _parse_datetime("") is None
        assert _parse_datetime("not-a-date") is None

    def test_first_float(self):
        from core.market_sources import _first_float
        assert _first_float(None, "", "0.5") == 0.5
        assert _first_float(None, None, default=0.0) == 0.0
        assert _first_float("0.75") == 0.75


# ---------------------------------------------------------------------------
# Model dataclass Tests
# ---------------------------------------------------------------------------
class TestModels:
    def test_prediction_outcome_as_dict(self):
        pred = PredictionOutcome(
            direction="Up",
            confidence=0.75,
            predicted_close=50100.0,
            last_close=50000.0,
            model_name="test-model",
        )
        d = pred.as_dict()
        assert d["direction"] == "Up"
        assert d["confidence"] == 0.75

    def test_market_opportunity(self):
        opp = MarketOpportunity(
            venue="polymarket",
            market_id="test-123",
            asset="BTC",
            share_price=0.65,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        assert opp.venue == "polymarket"
        assert opp.asset == "BTC"
