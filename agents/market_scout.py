from __future__ import annotations

from datetime import datetime, timezone

from agents.base import AgentContext, BaseAgent
from core.market_sources import scan_target_markets


class MarketScoutAgent(BaseAgent):
    def __init__(self, context: AgentContext) -> None:
        super().__init__(context, "market_scout")

    async def run(self):
        settings = self.context.settings
        return await scan_target_markets(
            min_minutes=settings.min_market_minutes,
            max_minutes=settings.max_market_minutes,
            assets=("BTC", "ETH"),
            now=datetime.now(timezone.utc),
            request_timeout=settings.request_timeout_seconds,
        )
