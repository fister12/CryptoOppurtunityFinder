"""Hermes Agent integration layer.

Uses the Hermes Agent SDK (from NousResearch/hermes-agent) when available,
with a direct OpenRouter fallback for environments where the full SDK
cannot be installed (e.g. CI, minimal containers).

The SDK provides: memory, skill learning, tool calling, and conversation
continuity — exactly what the assignment's "Hermes agents loop feedback"
requirement calls for.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the real Hermes Agent SDK
# ---------------------------------------------------------------------------
_HERMES_AVAILABLE = False
try:
    from run_agent import AIAgent  # type: ignore[import-untyped]
    _HERMES_AVAILABLE = True
    logger.info("Hermes Agent SDK loaded successfully")
except ImportError:
    logger.warning(
        "hermes-agent SDK not installed — falling back to direct OpenRouter. "
        "Install with: pip install git+https://github.com/NousResearch/hermes-agent.git"
    )


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class HermesAgentSpec:
    """Declarative spec for an LLM-backed agent role."""
    name: str
    system_prompt: str
    tool_names: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Central registry of callable tools exposed to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, handler: Callable[..., Awaitable[Any]]) -> None:
        self._tools[name] = ToolDefinition(name=name, description=description, handler=handler)

    def get(self, name: str) -> ToolDefinition:
        return self._tools[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)


# ---------------------------------------------------------------------------
# Hermes SDK wrapper (preferred path)
# ---------------------------------------------------------------------------
class HermesAgentClient:
    """Wraps the real Hermes Agent SDK ``AIAgent`` for library usage.

    Each trading agent (market scout, quant, risk, evaluator) gets its own
    ``AIAgent`` instance with ``quiet_mode=True`` so no TUI elements leak.
    """

    def __init__(self, model: str) -> None:
        if not _HERMES_AVAILABLE:
            raise ImportError("hermes-agent SDK is required but not installed")
        self.model = model
        self._agent = AIAgent(model=model, quiet_mode=True)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, int]]:
        """Send a conversation through the Hermes Agent and return (text, usage)."""
        # The SDK's .chat() is synchronous; wrap in asyncio.to_thread if needed
        import asyncio
        user_msg = messages[-1]["content"] if messages else ""

        # Prepend system prompt if provided in messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            context = system_msgs[0]["content"] + "\n\n" + user_msg
        else:
            context = user_msg

        response_text = await asyncio.to_thread(self._agent.chat, context)
        # SDK doesn't expose token counts directly; estimate from response
        usage = {
            "prompt_tokens": len(context.split()) * 2,  # rough estimate
            "completion_tokens": len(str(response_text).split()) * 2,
            "total_tokens": (len(context.split()) + len(str(response_text).split())) * 2,
        }
        return str(response_text), usage


# ---------------------------------------------------------------------------
# Direct OpenRouter fallback (works without the Hermes SDK installed)
# ---------------------------------------------------------------------------
class OpenRouterHermesClient:
    """Direct OpenRouter API client using the Hermes function-calling format.

    Uses the OpenAI-compatible chat completions API that OpenRouter exposes.
    This is the fallback when ``hermes-agent`` SDK is not installed.
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> tuple[str, dict[str, int]]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage is not None else {}
        return content, {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }


# ---------------------------------------------------------------------------
# Factory — automatically picks the best available backend
# ---------------------------------------------------------------------------
def create_llm_client(
    api_key: str,
    base_url: str,
    model: str,
    prefer_hermes: bool = True,
) -> OpenRouterHermesClient | HermesAgentClient:
    """Create the best available LLM client.

    Prefers the full Hermes Agent SDK when installed; falls back to direct
    OpenRouter otherwise.
    """
    if prefer_hermes and _HERMES_AVAILABLE:
        try:
            return HermesAgentClient(model=model)
        except Exception as exc:
            logger.warning("Failed to initialise Hermes SDK client: %s — falling back", exc)

    return OpenRouterHermesClient(api_key=api_key, base_url=base_url, model=model)
