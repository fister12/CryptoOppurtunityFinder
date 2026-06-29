from __future__ import annotations

from agents.base import AgentContext, BaseAgent
from core.risk import calculate_stake


class RiskManagerAgent(BaseAgent):
    def __init__(self, context: AgentContext) -> None:
        super().__init__(context, "risk_manager")

    async def run(self, opportunity, prediction):
        settings = self.context.settings
        return calculate_stake(
            bankroll_usd=settings.bankroll_usd,
            confidence=prediction.confidence,
            share_price=opportunity.share_price,
            fractional_kelly=settings.fractional_kelly,
        )
