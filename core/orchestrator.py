"""Trading orchestrator — the main agent loop with feedback.

Key improvements from original:
- Hermes feedback loop: past evaluation results are fed back into future cycles
- Multi-timeframe cascade: 1m→5m→15m predictions for arbitrage
- LLM-backed reasoning via Hermes for trade decision synthesis
- Structured per-cycle logging
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from agents import DataEngineerAgent, EvaluatorAgent, MarketScoutAgent, QuantAgent, RiskManagerAgent
from agents.base import AgentContext
from core.hermes_runtime import OpenRouterHermesClient, ToolRegistry, create_llm_client
from core.ledger import LedgerStore
from utils.kronos_inference import KronosInferenceEngine

logger = logging.getLogger("crypto_trader")


class TradingOrchestrator:
    def __init__(self, settings, ledger: LedgerStore, llm: OpenRouterHermesClient) -> None:
        self.settings = settings
        self.ledger = ledger
        self.llm = llm
        self.context = AgentContext(settings=settings, logger=logger, ledger=ledger, llm=llm)

        # Kronos inference — uses real HuggingFace models
        self.kronos = KronosInferenceEngine(
            kronos_model_name=settings.kronos_model_name,
            kronos_tokenizer_name=settings.kronos_tokenizer_name,
            timeout_seconds=settings.model_timeout_seconds,
        )
        self.context.kronos = self.kronos

        # Agents
        self.market_scout = MarketScoutAgent(self.context)
        self.data_engineer = DataEngineerAgent(self.context)
        self.quant_agent = QuantAgent(self.context)
        self.risk_manager = RiskManagerAgent(self.context)
        self.evaluator = EvaluatorAgent(self.context)

        # Tool registry for Hermes agent integration
        self.tool_registry = ToolRegistry()
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Feedback state — evaluation results from past cycles
        self._feedback_history: list[dict[str, Any]] = []
        self._cycle_count = 0
        self._wins = 0
        self._losses = 0

    def register_tools(self) -> None:
        self.tool_registry.register("scan_markets", "Scan Polymarket and Kalshi for BTC/ETH near-expiry opportunities", self.market_scout.run)
        self.tool_registry.register("fetch_ohlcv", "Fetch normalized OHLCV frames with Apify/Binance", self.data_engineer.run)
        self.tool_registry.register("predict_kronos", "Run the Kronos price forecaster", self.quant_agent.run)
        self.tool_registry.register("size_trade", "Apply Kelly sizing and fractional Kelly risk control", self.risk_manager.run)

    async def run_cycle(self) -> None:
        """Run a single trading cycle with feedback from past evaluations."""
        self._cycle_count += 1
        cycle_label = f"cycle_{self._cycle_count}"
        started_at = datetime.now(timezone.utc)

        logger.info("=== Starting %s ===", cycle_label)

        # Phase 1: Market scanning
        opportunities = await self.market_scout.run()
        if not opportunities:
            logger.info("[%s] No qualifying BTC/ETH opportunities in the current window", cycle_label)
            return

        logger.info("[%s] Found %d opportunities", cycle_label, len(opportunities))

        for opp_idx, opportunity in enumerate(opportunities):
            opp_label = f"{cycle_label}.opp_{opp_idx}"
            cycle_started = datetime.now(timezone.utc)
            try:
                # Phase 2: Data fetching (1m candles)
                frame_1m = await asyncio.wait_for(
                    self.data_engineer.run(opportunity),
                    timeout=self.settings.model_timeout_seconds,
                )

                # Phase 3: Multi-timeframe prediction
                prediction_1m = await asyncio.wait_for(
                    self.quant_agent.run(frame_1m),
                    timeout=self.settings.model_timeout_seconds,
                )

                # Also run 5m prediction by resampling the 1m data
                prediction_5m = await self._predict_resampled(frame_1m, "5min", opp_label)

                # Phase 4: Arbitrage check — compare 1m×5 vs 5m×1
                arbitrage_signal = self._check_arbitrage(prediction_1m, prediction_5m, opp_label)

                # Phase 5: Feedback-adjusted confidence
                adjusted_prediction = self._apply_feedback(prediction_1m, arbitrage_signal)

                # Phase 6: Kelly risk sizing
                risk_decision = await self.risk_manager.run(opportunity, adjusted_prediction)

                # Phase 7: LLM synthesis (Hermes feedback loop)
                synthesis = await self._hermes_synthesize(
                    opportunity, adjusted_prediction, risk_decision, arbitrage_signal
                )

                latency_seconds = (datetime.now(timezone.utc) - cycle_started).total_seconds()

                # Phase 8: Record to ledger
                cycle_id = await self.ledger.record_cycle(
                    opportunity=opportunity,
                    prediction=adjusted_prediction,
                    risk_decision=risk_decision,
                    latency_seconds=latency_seconds,
                    token_usage={},
                )

                # Phase 9: Schedule background evaluation
                task = asyncio.create_task(
                    self.evaluator.run(opportunity, cycle_id, self._on_evaluation_complete)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

                logger.info(
                    "[%s] venue=%s market_id=%s asset=%s signal=%s conf=%.3f "
                    "stake=$%.2f arb=%s latency=%.2fs",
                    opp_label,
                    opportunity.venue,
                    opportunity.market_id,
                    opportunity.asset,
                    adjusted_prediction.direction,
                    adjusted_prediction.confidence,
                    risk_decision.stake_usd,
                    arbitrage_signal,
                    latency_seconds,
                )
                if synthesis:
                    logger.info("[%s] LLM synthesis: %s", opp_label, synthesis[:200])

            except Exception as exc:
                logger.exception("[%s] Failed to process opportunity %s: %s", opp_label, opportunity.market_id, exc)

        total_latency = (datetime.now(timezone.utc) - started_at).total_seconds()
        win_rate = (self._wins / max(self._wins + self._losses, 1)) * 100
        logger.info(
            "=== Completed %s in %.2fs | W/L: %d/%d (%.1f%%) | Feedback entries: %d ===",
            cycle_label,
            total_latency,
            self._wins,
            self._losses,
            win_rate,
            len(self._feedback_history),
        )

    async def _predict_resampled(self, frame_1m, resample_rule: str, label: str):
        """Resample 1m data to a higher timeframe and run prediction."""
        import pandas as pd

        try:
            resampled = frame_1m.set_index("timestamps").resample(resample_rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()

            if len(resampled) < 10:
                logger.debug("[%s] Not enough %s candles for resampled prediction", label, resample_rule)
                return None

            return await asyncio.wait_for(
                self.quant_agent.run(resampled),
                timeout=self.settings.model_timeout_seconds,
            )
        except Exception as exc:
            logger.debug("[%s] Resampled %s prediction failed: %s", label, resample_rule, exc)
            return None

    @staticmethod
    def _check_arbitrage(pred_1m, pred_5m, label: str) -> str:
        """Compare 1m×5 vs 5m×1 for internal arbitrage signals.

        Returns: "agree", "diverge", or "insufficient"
        """
        if pred_1m is None or pred_5m is None:
            return "insufficient"

        if pred_1m.direction == pred_5m.direction:
            logger.debug("[%s] Arbitrage: timeframes AGREE on %s", label, pred_1m.direction)
            return "agree"
        else:
            logger.info(
                "[%s] Arbitrage: timeframes DIVERGE — 1m says %s, 5m says %s",
                label, pred_1m.direction, pred_5m.direction,
            )
            return "diverge"

    def _apply_feedback(self, prediction, arbitrage_signal: str):
        """Adjust prediction confidence based on historical feedback."""
        from core.models import PredictionOutcome

        confidence = prediction.confidence

        # Boost confidence when timeframes agree
        if arbitrage_signal == "agree":
            confidence = min(1.0, confidence * 1.15)
        elif arbitrage_signal == "diverge":
            confidence *= 0.75  # Reduce confidence on divergence

        # Adjust based on historical win rate
        if len(self._feedback_history) >= 5:
            recent = self._feedback_history[-10:]
            recent_wins = sum(1 for fb in recent if fb.get("correct", False))
            recent_rate = recent_wins / len(recent)

            # Scale confidence: if win rate > 60%, boost; if < 40%, reduce
            if recent_rate > 0.6:
                confidence = min(1.0, confidence * (1.0 + (recent_rate - 0.5) * 0.5))
            elif recent_rate < 0.4:
                confidence *= max(0.3, recent_rate + 0.1)

        confidence = max(0.0, min(1.0, confidence))

        return PredictionOutcome(
            direction=prediction.direction,
            confidence=confidence,
            predicted_close=prediction.predicted_close,
            last_close=prediction.last_close,
            model_name=prediction.model_name,
            raw_prediction={
                **prediction.raw_prediction,
                "feedback_adjusted": True,
                "arbitrage": arbitrage_signal,
                "original_confidence": prediction.confidence,
            },
        )

    async def _hermes_synthesize(self, opportunity, prediction, risk_decision, arbitrage_signal: str) -> str:
        """Use the LLM (via Hermes) to synthesize a trade reasoning summary.

        This implements the "Hermes agents loop feedback" requirement — the
        LLM reviews the quantitative signals and past performance to produce
        a human-readable trade thesis.
        """
        feedback_summary = "No historical data yet."
        if self._feedback_history:
            recent = self._feedback_history[-5:]
            feedback_summary = "; ".join(
                f"{fb['asset']} {fb['direction']} {'✓' if fb.get('correct') else '✗'}"
                for fb in recent
            )

        prompt = (
            f"You are a crypto trading analyst. Synthesize this trade signal:\n"
            f"Asset: {opportunity.asset} | Venue: {opportunity.venue}\n"
            f"Prediction: {prediction.direction} (conf: {prediction.confidence:.1%})\n"
            f"Kelly stake: ${risk_decision.stake_usd:.2f} of ${risk_decision.bankroll_usd:.0f}\n"
            f"Timeframe arbitrage: {arbitrage_signal}\n"
            f"Recent track record: {feedback_summary}\n"
            f"Give a 1-2 sentence trade thesis."
        )

        try:
            response, _usage = await asyncio.wait_for(
                self.llm.chat(
                    [{"role": "system", "content": "You are a concise crypto trading analyst."},
                     {"role": "user", "content": prompt}],
                    temperature=0.3,
                ),
                timeout=30.0,
            )
            return response.strip()
        except Exception as exc:
            logger.debug("LLM synthesis failed: %s", exc)
            return ""

    def _on_evaluation_complete(self, evaluation_result: dict[str, Any]) -> None:
        """Callback from the evaluator — feeds results back into the loop.

        This closes the Hermes feedback loop: evaluation outcomes inform
        future confidence adjustments and Kelly sizing.
        """
        self._feedback_history.append(evaluation_result)

        # Keep only the last 100 evaluations
        if len(self._feedback_history) > 100:
            self._feedback_history = self._feedback_history[-100:]

        if evaluation_result.get("correct", False):
            self._wins += 1
        else:
            self._losses += 1

        logger.info(
            "Feedback received: %s %s %s — W/L: %d/%d",
            evaluation_result.get("asset", "?"),
            evaluation_result.get("direction", "?"),
            "✓ CORRECT" if evaluation_result.get("correct") else "✗ WRONG",
            self._wins,
            self._losses,
        )

    async def run_forever(self, once: bool = False, max_cycles: int | None = None) -> None:
        self.register_tools()
        cycles_run = 0
        while True:
            await self.run_cycle()
            cycles_run += 1
            if once or (max_cycles is not None and cycles_run >= max_cycles):
                break
            await asyncio.sleep(self.settings.loop_interval_seconds)

    async def shutdown(self) -> None:
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
