from __future__ import annotations

from agents.base import AgentContext, BaseAgent


class QuantAgent(BaseAgent):
    def __init__(self, context: AgentContext) -> None:
        super().__init__(context, "quant_agent")

    async def run(self, frame):
        if self.context.kronos is None:
            raise RuntimeError("Kronos inference engine is not configured")
        return await self.context.kronos.predict(frame)
