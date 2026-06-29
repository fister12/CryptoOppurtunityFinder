"""Kronos model integration — proper wrapper around the real Kronos SDK.

Kronos (shiyu-coder/Kronos) is a foundation model for financial K-lines.
It uses a specialized tokenizer (KronosTokenizer) that quantises OHLCV
candlestick data into hierarchical discrete tokens, then a decoder-only
transformer predicts future candles.

This module handles:
  1. Dynamically loading the Kronos ``model`` package from the cloned repo
  2. Providing ``load_kronos_predictor()`` to get a ready-to-use predictor
  3. OHLCV DataFrame normalisation for the Kronos input format
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic import of the Kronos model package from cloned repo
# ---------------------------------------------------------------------------
_KRONOS_AVAILABLE = False
_KronosTokenizer = None
_Kronos = None
_KronosPredictor = None

def _try_import_kronos() -> bool:
    """Attempt to import Kronos model classes from the cloned repo."""
    global _KRONOS_AVAILABLE, _KronosTokenizer, _Kronos, _KronosPredictor

    if _KRONOS_AVAILABLE:
        return True

    # Look for the cloned repo in project root
    kronos_repo = Path(__file__).parent.parent / "kronos_repo"
    if kronos_repo.exists() and str(kronos_repo) not in sys.path:
        sys.path.insert(0, str(kronos_repo))
        logger.info("Added Kronos repo to sys.path: %s", kronos_repo)

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        _KronosTokenizer = KronosTokenizer
        _Kronos = Kronos
        _KronosPredictor = KronosPredictor
        _KRONOS_AVAILABLE = True
        logger.info("Kronos model classes loaded successfully")
        return True
    except ImportError as exc:
        logger.warning(
            "Kronos model not available: %s. "
            "Run 'python setup_kronos.py' to clone the repo.",
            exc,
        )
        return False


def is_kronos_available() -> bool:
    """Check if the Kronos model code is importable."""
    return _try_import_kronos()


# ---------------------------------------------------------------------------
# OHLCV Data Normalisation
# ---------------------------------------------------------------------------
def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise an OHLCV DataFrame for Kronos consumption.

    Expected columns: open, high, low, close, volume, timestamps.
    Returns a sorted, deduplicated, NaN-free DataFrame.
    """
    required = ["open", "high", "low", "close", "volume", "timestamps"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = frame.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamps")
        .drop_duplicates(subset=["timestamps"])
        .tail(1000)
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No usable candles after normalisation")
    return df


def frame_to_ohlcv_array(frame: pd.DataFrame) -> np.ndarray:
    """Convert a normalised OHLCV DataFrame into a numpy array for Kronos.

    Returns shape (N, 5) with columns [open, high, low, close, volume].
    """
    return frame[["open", "high", "low", "close", "volume"]].values.astype(np.float32)


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics from an OHLCV DataFrame."""
    df = normalize_ohlcv_frame(frame)
    last_close = float(df.iloc[-1]["close"])
    return {
        "last_close": last_close,
        "mean_close": float(df["close"].mean()),
        "volatility": float(df["close"].pct_change().fillna(0.0).std()),
        "row_count": len(df),
        "start_timestamp": df.iloc[0]["timestamps"].isoformat(),
        "end_timestamp": df.iloc[-1]["timestamps"].isoformat(),
    }


def infer_direction(predicted_close: float, last_close: float) -> str:
    """Simple up/down direction from predicted vs last close."""
    return "Up" if predicted_close >= last_close else "Down"
