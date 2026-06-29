from __future__ import annotations

from core.models import RiskDecision


def _decimal_odds_from_share_price(share_price: float) -> float:
    if share_price <= 0.0 or share_price >= 1.0:
        return 0.0
    return (1.0 - share_price) / share_price


def calculate_kelly_fraction(confidence: float, share_price: float) -> float:
    confidence = max(0.0, min(1.0, confidence))
    decimal_odds = _decimal_odds_from_share_price(share_price)
    if decimal_odds <= 0.0:
        return 0.0

    kelly_fraction = confidence - ((1.0 - confidence) / decimal_odds)
    if kelly_fraction < 0.01:
        return 0.0
    return max(0.0, kelly_fraction)


def calculate_stake(
    bankroll_usd: float,
    confidence: float,
    share_price: float,
    fractional_kelly: float = 0.5,
) -> RiskDecision:
    raw_fraction = calculate_kelly_fraction(confidence, share_price)
    fractional_fraction = raw_fraction * max(0.0, fractional_kelly)
    stake_usd = 0.0 if fractional_fraction < 0.01 else round(bankroll_usd * fractional_fraction, 2)

    return RiskDecision(
        bankroll_usd=bankroll_usd,
        share_price=share_price,
        confidence=confidence,
        raw_kelly_fraction=raw_fraction,
        fractional_kelly_fraction=fractional_fraction,
        stake_usd=stake_usd,
    )
