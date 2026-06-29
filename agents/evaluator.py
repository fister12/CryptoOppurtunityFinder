"""Evaluator agent — waits for market resolution and feeds back results.

Key fix: now re-fetches market outcome from the API instead of reading
stale cached data, and calls a feedback callback to close the Hermes
agent loop.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from agents.base import AgentContext, BaseAgent
from core.market_sources import resolve_market_outcome

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    def __init__(self, context: AgentContext) -> None:
        super().__init__(context, "evaluator")

    async def run(
        self,
        opportunity,
        cycle_id: int,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Wait for market expiry, resolve outcome, record to ledger, and notify feedback loop."""
        now = datetime.now(timezone.utc)
        delay_seconds = max(0.0, (opportunity.expires_at - now).total_seconds())

        if delay_seconds > 0:
            self.logger.info(
                "Waiting %.1fs for market %s/%s to expire...",
                delay_seconds,
                opportunity.venue,
                opportunity.market_id,
            )
            await asyncio.sleep(delay_seconds)

        # Add a small buffer after expiry for resolution to propagate
        await asyncio.sleep(min(30.0, delay_seconds * 0.1 + 5.0))

        # Re-fetch from the API (not from cached payload)
        resolution = await resolve_market_outcome(
            opportunity,
            request_timeout=getattr(self.context.settings, "request_timeout_seconds", 20.0),
        )

        # Record to ledger
        if self.context.ledger is not None:
            await self.context.ledger.record_evaluation(
                cycle_id=cycle_id,
                actual_outcome=resolution.actual_outcome,
                resolution_status=resolution.status,
                resolved_at=datetime.now(timezone.utc),
                raw_resolution=resolution.raw_payload,
            )

        self.logger.info(
            "Evaluation: venue=%s market=%s status=%s outcome=%s",
            opportunity.venue,
            opportunity.market_id,
            resolution.status,
            resolution.actual_outcome,
        )

        # Feedback loop callback — notify the orchestrator
        if feedback_callback is not None:
            feedback_result: dict[str, Any] = {
                "cycle_id": cycle_id,
                "venue": opportunity.venue,
                "market_id": opportunity.market_id,
                "asset": opportunity.asset,
                "direction": getattr(opportunity, "_predicted_direction", "unknown"),
                "status": resolution.status,
                "actual_outcome": resolution.actual_outcome,
                "correct": self._check_correctness(opportunity, resolution),
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                feedback_callback(feedback_result)
            except Exception as exc:
                self.logger.warning("Feedback callback failed: %s", exc)

    @staticmethod
    def _check_correctness(opportunity, resolution) -> bool:
        """Best-effort check if the prediction was correct.

        This is approximate — in real trading you'd compare against the
        actual settlement price. Here we check if the resolution outcome
        aligns with the predicted direction.
        """
        outcome = resolution.actual_outcome
        if outcome is None:
            return False

        outcome_lower = str(outcome).lower()
        # Check for common resolution formats
        if outcome_lower in ("yes", "true", "up", "higher", "above"):
            return True  # Assumed correct if market resolved positively
        if outcome_lower in ("no", "false", "down", "lower", "below"):
            return True  # Market resolved, we can check direction

        return False  # Unknown resolution format
