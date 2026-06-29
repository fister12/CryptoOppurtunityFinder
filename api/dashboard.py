"""FastAPI dashboard — user visibility into the trading system.

Exposes REST endpoints for:
  - Recent trading cycles and their outcomes
  - Win/loss statistics
  - Ledger query
  - Health check

Start with: python main.py --api
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from core.ledger import LedgerStore


def create_app(ledger: LedgerStore) -> FastAPI:
    app = FastAPI(
        title="Crypto Trading Agents Dashboard",
        description="Real-time visibility into the Hermes-powered trading system",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/api/cycles")
    async def get_cycles(limit: int = 20):
        """Get recent trading cycles."""
        if ledger._connection is None:
            return {"cycles": [], "error": "ledger not connected"}

        cursor = await ledger._connection.execute(
            "SELECT * FROM trading_cycles ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        cycles = []
        for row in rows:
            cycle = dict(row)
            if "raw_payload" in cycle and isinstance(cycle["raw_payload"], str):
                try:
                    cycle["raw_payload"] = json.loads(cycle["raw_payload"])
                except json.JSONDecodeError:
                    pass
            cycles.append(cycle)

        return {"cycles": cycles, "count": len(cycles)}

    @app.get("/api/evaluations")
    async def get_evaluations(limit: int = 20):
        """Get recent evaluations."""
        if ledger._connection is None:
            return {"evaluations": [], "error": "ledger not connected"}

        cursor = await ledger._connection.execute(
            "SELECT * FROM trading_evaluations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        evaluations = []
        for row in rows:
            ev = dict(row)
            if "raw_resolution" in ev and isinstance(ev["raw_resolution"], str):
                try:
                    ev["raw_resolution"] = json.loads(ev["raw_resolution"])
                except json.JSONDecodeError:
                    pass
            evaluations.append(ev)

        return {"evaluations": evaluations, "count": len(evaluations)}

    @app.get("/api/stats")
    async def get_stats():
        """Get aggregate trading statistics."""
        if ledger._connection is None:
            return {"error": "ledger not connected"}

        cursor = await ledger._connection.execute("SELECT COUNT(*) as total FROM trading_cycles")
        row = await cursor.fetchone()
        total_cycles = row["total"] if row else 0

        cursor = await ledger._connection.execute("SELECT COUNT(*) as total FROM trading_evaluations")
        row = await cursor.fetchone()
        total_evals = row["total"] if row else 0

        cursor = await ledger._connection.execute("SELECT AVG(confidence) as avg_conf FROM trading_cycles")
        row = await cursor.fetchone()
        avg_confidence = round(row["avg_conf"], 4) if row and row["avg_conf"] else 0.0

        cursor = await ledger._connection.execute("SELECT SUM(stake_usd) as total_stake FROM trading_cycles")
        row = await cursor.fetchone()
        total_stake = round(row["total_stake"], 2) if row and row["total_stake"] else 0.0

        cursor = await ledger._connection.execute("SELECT AVG(latency_seconds) as avg_lat FROM trading_cycles")
        row = await cursor.fetchone()
        avg_latency = round(row["avg_lat"], 2) if row and row["avg_lat"] else 0.0

        cursor = await ledger._connection.execute(
            "SELECT direction, COUNT(*) as cnt FROM trading_cycles GROUP BY direction"
        )
        rows = await cursor.fetchall()
        direction_dist = {row["direction"]: row["cnt"] for row in rows}

        cursor = await ledger._connection.execute(
            "SELECT asset, COUNT(*) as cnt FROM trading_cycles GROUP BY asset"
        )
        rows = await cursor.fetchall()
        asset_dist = {row["asset"]: row["cnt"] for row in rows}

        return {
            "total_cycles": total_cycles,
            "total_evaluations": total_evals,
            "avg_confidence": avg_confidence,
            "total_stake_usd": total_stake,
            "avg_latency_seconds": avg_latency,
            "direction_distribution": direction_dist,
            "asset_distribution": asset_dist,
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Simple HTML dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Agents Dashboard</title>
            <style>
                body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }
                h1 { color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
                .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }
                .stat { display: inline-block; margin: 8px 16px; }
                .stat-value { font-size: 24px; font-weight: bold; color: #58a6ff; }
                .stat-label { font-size: 12px; color: #8b949e; }
                table { width: 100%; border-collapse: collapse; margin-top: 12px; }
                th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #21262d; }
                th { color: #8b949e; font-weight: 600; }
                .up { color: #3fb950; } .down { color: #f85149; }
                #loading { color: #8b949e; }
            </style>
        </head>
        <body>
            <h1>&#x1F916; Crypto Trading Agents Dashboard</h1>
            <div class="card" id="stats-card"><div id="loading">Loading stats...</div></div>
            <h2>Recent Cycles</h2>
            <div class="card"><table id="cycles-table"><thead><tr>
                <th>ID</th><th>Time</th><th>Venue</th><th>Asset</th><th>Signal</th>
                <th>Confidence</th><th>Stake</th><th>Latency</th>
            </tr></thead><tbody id="cycles-body"></tbody></table></div>
            <script>
                async function load() {
                    const [statsRes, cyclesRes] = await Promise.all([
                        fetch('/api/stats').then(r => r.json()),
                        fetch('/api/cycles').then(r => r.json())
                    ]);
                    document.getElementById('stats-card').innerHTML =
                        '<div class="stat"><div class="stat-value">' + statsRes.total_cycles + '</div><div class="stat-label">Total Cycles</div></div>' +
                        '<div class="stat"><div class="stat-value">' + (statsRes.avg_confidence * 100).toFixed(1) + '%</div><div class="stat-label">Avg Confidence</div></div>' +
                        '<div class="stat"><div class="stat-value">$' + statsRes.total_stake_usd + '</div><div class="stat-label">Total Stake</div></div>' +
                        '<div class="stat"><div class="stat-value">' + statsRes.avg_latency_seconds + 's</div><div class="stat-label">Avg Latency</div></div>' +
                        '<div class="stat"><div class="stat-value">' + statsRes.total_evaluations + '</div><div class="stat-label">Evaluations</div></div>';
                    var tbody = document.getElementById('cycles-body');
                    tbody.innerHTML = cyclesRes.cycles.map(function(c) {
                        return '<tr><td>' + c.id + '</td><td>' + (c.created_at || '').slice(0,19) + '</td><td>' + c.venue + '</td><td>' + c.asset + '</td>' +
                            '<td class="' + (c.direction === 'Up' ? 'up' : 'down') + '">' + c.direction + '</td>' +
                            '<td>' + (c.confidence * 100).toFixed(1) + '%</td><td>$' + c.stake_usd + '</td>' +
                            '<td>' + (c.latency_seconds || 0).toFixed(2) + 's</td></tr>';
                    }).join('');
                }
                load(); setInterval(load, 30000);
            </script>
        </body>
        </html>
        """

    return app
