from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MarketOpportunity:
    venue: str
    market_id: str
    asset: str
    share_price: float
    expires_at: datetime
    title: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionOutcome:
    direction: str
    confidence: float
    predicted_close: float
    last_close: float
    model_name: str
    raw_prediction: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RiskDecision:
    bankroll_usd: float
    share_price: float
    confidence: float
    raw_kelly_fraction: float
    fractional_kelly_fraction: float
    stake_usd: float


@dataclass(slots=True)
class ResolutionResult:
    status: str
    actual_outcome: str | None
    raw_payload: dict[str, Any] = field(default_factory=dict)
