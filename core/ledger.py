from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


class LedgerStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self._connection: aiosqlite.Connection | None = None

    @classmethod
    async def create(cls, sqlite_path: Path) -> "LedgerStore":
        ledger = cls(sqlite_path)
        await ledger._initialize()
        return ledger

    async def _initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.sqlite_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trading_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                venue TEXT NOT NULL,
                market_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                title TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                share_price REAL NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_close REAL NOT NULL,
                last_close REAL NOT NULL,
                bankroll_usd REAL NOT NULL,
                raw_kelly_fraction REAL NOT NULL,
                fractional_kelly_fraction REAL NOT NULL,
                stake_usd REAL NOT NULL,
                openrouter_prompt_tokens INTEGER,
                openrouter_completion_tokens INTEGER,
                openrouter_total_tokens INTEGER,
                latency_seconds REAL NOT NULL,
                raw_payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trading_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                resolved_at TEXT NOT NULL,
                actual_outcome TEXT,
                resolution_status TEXT NOT NULL,
                raw_resolution TEXT NOT NULL,
                FOREIGN KEY (cycle_id) REFERENCES trading_cycles(id)
            );
            """
        )
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def record_cycle(
        self,
        *,
        opportunity,
        prediction,
        risk_decision,
        latency_seconds: float,
        token_usage: dict[str, Any] | None = None,
    ) -> int:
        if self._connection is None:
            raise RuntimeError("ledger is not initialized")

        token_usage = token_usage or {}
        cursor = await self._connection.execute(
            """
            INSERT INTO trading_cycles (
                created_at, venue, market_id, asset, title, expires_at, share_price,
                direction, confidence, predicted_close, last_close,
                bankroll_usd, raw_kelly_fraction, fractional_kelly_fraction, stake_usd,
                openrouter_prompt_tokens, openrouter_completion_tokens, openrouter_total_tokens,
                latency_seconds, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                opportunity.venue,
                opportunity.market_id,
                opportunity.asset,
                opportunity.title,
                opportunity.expires_at.isoformat(),
                opportunity.share_price,
                prediction.direction,
                prediction.confidence,
                prediction.predicted_close,
                prediction.last_close,
                risk_decision.bankroll_usd,
                risk_decision.raw_kelly_fraction,
                risk_decision.fractional_kelly_fraction,
                risk_decision.stake_usd,
                int(token_usage.get("prompt_tokens") or 0),
                int(token_usage.get("completion_tokens") or 0),
                int(token_usage.get("total_tokens") or 0),
                latency_seconds,
                json.dumps(opportunity.raw_payload, default=str),
            ),
        )
        await self._connection.commit()
        return int(cursor.lastrowid)

    async def record_evaluation(
        self,
        *,
        cycle_id: int,
        actual_outcome: str | None,
        resolution_status: str,
        resolved_at: datetime,
        raw_resolution: dict[str, Any],
    ) -> None:
        if self._connection is None:
            raise RuntimeError("ledger is not initialized")

        await self._connection.execute(
            """
            INSERT INTO trading_evaluations (
                cycle_id, resolved_at, actual_outcome, resolution_status, raw_resolution
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                cycle_id,
                resolved_at.isoformat(),
                actual_outcome,
                resolution_status,
                json.dumps(raw_resolution, default=str),
            ),
        )
        await self._connection.commit()
