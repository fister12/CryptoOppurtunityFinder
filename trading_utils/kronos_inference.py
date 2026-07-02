"""Kronos inference engine — runs predictions using the real Kronos SDK.

This replaces the broken AutoTokenizer/AutoModelForCausalLM approach.
Kronos processes raw OHLCV candle data (not text), quantises it through
its specialised tokenizer, and autoregressively predicts future candles.

When the Kronos SDK is not available (repo not cloned), it falls back to
a simple statistical heuristic (momentum + mean-reversion) so the system
can still produce predictions for testing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.models import PredictionOutcome
from models.kronos import (
    _KronosPredictor,
    _KronosTokenizer,
    _Kronos,
    frame_to_ohlcv_array,
    infer_direction,
    is_kronos_available,
    normalize_ohlcv_frame,
    summarize_frame,
)

logger = logging.getLogger(__name__)


class KronosInferenceEngine:
    """Runs next-candle predictions using the Kronos foundation model.

    Falls back to a statistical heuristic when the Kronos SDK is unavailable.
    """

    def __init__(
        self,
        kronos_model_name: str = "NeoQuasar/Kronos-mini",
        kronos_tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.kronos_model_name = kronos_model_name
        self.kronos_tokenizer_name = kronos_tokenizer_name
        self.timeout_seconds = timeout_seconds
        self._predictor = None
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Lazy-load the Kronos model from HuggingFace on first use."""
        if self._loaded:
            return

        if not is_kronos_available():
            logger.warning(
                "Kronos SDK not available — using statistical fallback. "
                "Run 'python setup_kronos.py' to enable Kronos."
            )
            self._loaded = True
            return

        logger.info(
            "Loading Kronos model=%s tokenizer=%s (this may download ~50MB on first run)...",
            self.kronos_model_name,
            self.kronos_tokenizer_name,
        )
        await asyncio.wait_for(
            asyncio.to_thread(self._load_sync),
            timeout=self.timeout_seconds,
        )
        self._loaded = True

    def _load_sync(self) -> None:
        """Synchronous model loading (runs in a thread)."""
        tokenizer = _KronosTokenizer.from_pretrained(self.kronos_tokenizer_name)
        model = _Kronos.from_pretrained(self.kronos_model_name)
        model.eval()

        # Kronos-mini supports max_context=2048, Kronos-small/base support 512
        max_ctx = 2048 if "mini" in self.kronos_model_name.lower() else 512
        self._predictor = _KronosPredictor(model, tokenizer, max_context=max_ctx)
        logger.info("Kronos model loaded successfully (max_context=%d)", max_ctx)

    async def predict(self, frame: pd.DataFrame) -> PredictionOutcome:
        """Predict the next candle's close price direction.

        Uses Kronos when available, otherwise falls back to a momentum
        + mean-reversion heuristic.
        """
        await self._ensure_loaded()

        # Normalise and summarise
        normalized = normalize_ohlcv_frame(frame)
        stats = summarize_frame(frame)
        last_close = stats["last_close"]

        if self._predictor is not None:
            # ---------- Real Kronos path ----------
            predicted_close, raw = await asyncio.wait_for(
                asyncio.to_thread(self._kronos_predict_sync, normalized),
                timeout=self.timeout_seconds,
            )
        else:
            # ---------- Statistical fallback ----------
            predicted_close, raw = self._statistical_fallback(normalized, last_close)

        direction = infer_direction(predicted_close, last_close)
        movement = abs(predicted_close - last_close) / max(last_close, 1e-9)
        confidence = max(0.0, min(1.0, 0.5 + movement * 5.0))

        return PredictionOutcome(
            direction=direction,
            confidence=confidence,
            predicted_close=predicted_close,
            last_close=last_close,
            model_name=self.kronos_model_name,
            raw_prediction={**raw, "confidence": confidence},
        )

    def _kronos_predict_sync(self, normalized: pd.DataFrame) -> tuple[float, dict[str, Any]]:
        """Run Kronos prediction synchronously (called from thread)."""
        ohlcv = frame_to_ohlcv_array(normalized)

        with torch.inference_mode():
            # KronosPredictor.predict expects OHLCV numpy arrays
            # and returns predicted future candles
            prediction = self._predictor.predict(
                ohlcv,
                horizon=5,  # predict next 5 candles (5-min lookahead)
            )

        # prediction shape: (horizon, 5) — [open, high, low, close, volume]
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()

        predicted_close = float(prediction[-1, 3])  # close column of last predicted candle
        raw: dict[str, Any] = {
            "mode": "kronos",
            "model": self.kronos_model_name,
            "predicted_candles": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
        }
        return predicted_close, raw

    @staticmethod
    def _statistical_fallback(
        normalized: pd.DataFrame,
        last_close: float,
    ) -> tuple[float, dict[str, Any]]:
        """Momentum + mean-reversion heuristic as fallback when Kronos unavailable.

        Combines:
        - Short-term momentum (last 5 candles)
        - Mean-reversion signal towards the 50-candle SMA
        - Volatility scaling
        """
        closes = normalized["close"].values.astype(np.float64)

        # Short-term momentum: weighted average of recent returns
        recent = closes[-min(5, len(closes)):]
        if len(recent) > 1:
            returns = np.diff(recent) / recent[:-1]
            momentum = float(np.mean(returns))
        else:
            momentum = 0.0

        # Mean-reversion towards 50-candle SMA
        sma_window = min(50, len(closes))
        sma = float(np.mean(closes[-sma_window:]))
        mean_rev_signal = (sma - last_close) / max(last_close, 1e-9)

        # Blend: 70% momentum, 30% mean-reversion
        blended_return = 0.7 * momentum + 0.3 * mean_rev_signal * 0.1

        # Clamp to ±5%
        blended_return = max(-0.05, min(0.05, blended_return))
        predicted_close = last_close * (1.0 + blended_return)

        raw: dict[str, Any] = {
            "mode": "statistical_fallback",
            "momentum": momentum,
            "mean_reversion": mean_rev_signal,
            "blended_return": blended_return,
        }
        return predicted_close, raw
