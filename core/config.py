from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during import
    load_dotenv = None


@dataclass(slots=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    apify_api_token: str | None
    apify_actor_id: str
    bankroll_usd: float
    loop_interval_seconds: int
    min_market_minutes: int
    max_market_minutes: int
    model_timeout_seconds: float
    sqlite_path: Path
    request_timeout_seconds: float
    max_retries: int
    fractional_kelly: float
    kronos_model_name: str
    kronos_tokenizer_name: str


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def load_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv()

    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        apify_api_token=os.getenv("APIFY_API_TOKEN") or None,
        apify_actor_id=os.getenv("APIFY_CRYPTO_ACTOR_ID", "apify/crypto-ohlcv-scraper"),
        bankroll_usd=_read_float("TRADING_BANKROLL_USD", 1000.0),
        loop_interval_seconds=_read_int("TRADING_LOOP_INTERVAL_SECONDS", 300),
        min_market_minutes=_read_int("TRADING_MIN_EXPIRY_MINUTES", 5),
        max_market_minutes=_read_int("TRADING_MAX_EXPIRY_MINUTES", 15),
        model_timeout_seconds=_read_float("TRADING_MODEL_TIMEOUT_SECONDS", 120.0),
        sqlite_path=Path(os.getenv("TRADING_SQLITE_PATH", "ledger/trading_ledger.sqlite3")),
        request_timeout_seconds=_read_float("TRADING_REQUEST_TIMEOUT_SECONDS", 20.0),
        max_retries=_read_int("TRADING_MAX_RETRIES", 3),
        fractional_kelly=_read_float("TRADING_FRACTIONAL_KELLY", 0.5),
        kronos_model_name=os.getenv("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-mini"),
        kronos_tokenizer_name=os.getenv("KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-2k"),
    )
