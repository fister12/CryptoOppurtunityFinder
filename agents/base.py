from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger


@dataclass(slots=True)
class AgentContext:
    settings: object
    logger: Logger
    ledger: object | None = None
    llm: object | None = None
    kronos: object | None = None


class BaseAgent(ABC):
    def __init__(self, context: AgentContext, name: str) -> None:
        self.context = context
        self.name = name
        self.logger = context.logger.getChild(name)

    @abstractmethod
    async def run(self, *args, **kwargs):
        raise NotImplementedError
