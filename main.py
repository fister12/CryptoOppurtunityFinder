"""Entry point for the crypto trading agent system.

Supports:
  --once          Run a single cycle and exit
  --max-cycles N  Limit cycles before exiting
  --api           Start the FastAPI dashboard alongside the trading loop
  --log-level     Set logging verbosity (default: INFO)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import structlog

from core.config import load_settings
from core.hermes_runtime import create_llm_client
from core.ledger import LedgerStore
from core.orchestrator import TradingOrchestrator


def _configure_logging(level: str) -> None:
    """Set up structured logging with structlog + stdlib integration."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes-based crypto prediction trading backend")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--max-cycles", type=int, default=None, help="Limit the number of cycles before exiting")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI dashboard API alongside trading")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for the dashboard API (default: 8000)")
    return parser


async def async_main(
    once: bool = False,
    max_cycles: int | None = None,
    run_api: bool = False,
    api_port: int = 8000,
) -> None:
    settings = load_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required — set it in .env")

    ledger = await LedgerStore.create(settings.sqlite_path)

    # Use the factory to pick the best available LLM backend
    llm = create_llm_client(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
    )
    logging.getLogger("crypto_trader").info(
        "LLM backend: %s (model=%s)", type(llm).__name__, settings.openrouter_model
    )

    orchestrator = TradingOrchestrator(settings=settings, ledger=ledger, llm=llm)

    api_task = None
    if run_api:
        from api.dashboard import create_app
        import uvicorn

        app = create_app(ledger)
        config = uvicorn.Config(app, host="0.0.0.0", port=api_port, log_level="info")
        server = uvicorn.Server(config)
        api_task = asyncio.create_task(server.serve())
        logging.getLogger("crypto_trader").info("Dashboard API started on http://0.0.0.0:%d", api_port)

    try:
        await orchestrator.run_forever(once=once, max_cycles=max_cycles)
    finally:
        await orchestrator.shutdown()
        await ledger.close()
        if api_task is not None:
            api_task.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        asyncio.run(async_main(
            once=args.once,
            max_cycles=args.max_cycles,
            run_api=args.api,
            api_port=args.api_port,
        ))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
