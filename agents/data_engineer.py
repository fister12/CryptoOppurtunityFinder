from __future__ import annotations

from agents.base import AgentContext, BaseAgent
from utils.apify_client import ApifyOHLCVClient


class DataEngineerAgent(BaseAgent):
    def __init__(self, context: AgentContext) -> None:
        super().__init__(context, "data_engineer")
        self.client = ApifyOHLCVClient(context.settings)

    async def run(self, opportunity):
        return await self.client.fetch_ohlcv(
            asset=opportunity.asset,
            limit=1000,
            timeframe="1m",
        )
