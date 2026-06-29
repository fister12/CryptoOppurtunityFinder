# Crypto Trading Agents 🤖

**Hermes-powered autonomous crypto prediction trading system** targeting Polymarket and Kalshi binary options for BTC and ETH.

Built with the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework and [Kronos](https://github.com/shiyu-coder/Kronos) financial market foundation model.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Market Scout │────▶│Data Engineer │────▶│  Quant Agent │
│  (Poly+Kal)  │     │(Apify+Binance│     │   (Kronos)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                          ┌───────────────────────┘
                          ▼
                  ┌──────────────┐     ┌──────────────┐
                  │Risk Manager  │────▶│  Evaluator   │
                  │(Kelly Crit.) │     │ (Feedback)   │
                  └──────────────┘     └──────┬───────┘
                                              │
                          ┌───────────────────┘
                          ▼
                  ┌──────────────────────────────────┐
                  │    Hermes Feedback Loop           │
                  │  Past W/L → Confidence Scaling   │
                  │  Multi-TF Arbitrage Signals       │
                  │  LLM Trade Thesis Synthesis       │
                  └──────────────────────────────────┘
```

### Agent Roles

| Agent | Role |
|-------|------|
| **Market Scout** | Scans Polymarket + Kalshi for BTC/ETH binary options expiring in 5–15 min |
| **Data Engineer** | Fetches 1000 OHLCV candles via Apify (with Binance public API fallback) |
| **Quant Agent** | Runs [Kronos](https://github.com/shiyu-coder/Kronos) foundation model for next-candle prediction |
| **Risk Manager** | Applies Kelly Criterion with fractional sizing for position management |
| **Evaluator** | Re-fetches market outcomes after expiry, feeds results back into the loop |

### Scaling Features

- **Multi-timeframe prediction**: 1m candles resampled to 5m for cross-timeframe validation
- **Internal arbitrage**: Detects divergence between 1m×5 and 5m×1 predictions
- **Hermes feedback loop**: Win/loss history adjusts future confidence scoring
- **LLM trade synthesis**: Hermes Agent generates human-readable trade theses

## Quick Start

### 1. Clone and setup

```bash
git clone <repo-url>
cd CryptoAssing

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kronos model

```bash
python setup_kronos.py
```

This clones the Kronos repo. Models (~50MB) are auto-downloaded from HuggingFace on first run.

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env with your API keys:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   APIFY_API_TOKEN=apify_api_...  (optional, Binance fallback available)
```

### 4. Run

```bash
# Single cycle (test)
python main.py --once --log-level DEBUG

# Continuous trading loop
python main.py

# With dashboard API
python main.py --api --api-port 8000
```

### 5. View dashboard

Open http://localhost:8000 when running with `--api`.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML dashboard |
| `GET /health` | Health check |
| `GET /api/cycles?limit=20` | Recent trading cycles |
| `GET /api/evaluations?limit=20` | Recent evaluations |
| `GET /api/stats` | Aggregate statistics |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key |
| `OPENROUTER_MODEL` | `google/gemma-2-9b-it:free` | LLM model for reasoning |
| `APIFY_API_TOKEN` | (optional) | Apify token; falls back to Binance |
| `KRONOS_MODEL_NAME` | `NeoQuasar/Kronos-mini` | Kronos model from HuggingFace |
| `KRONOS_TOKENIZER_NAME` | `NeoQuasar/Kronos-Tokenizer-2k` | Kronos tokenizer |
| `TRADING_BANKROLL_USD` | `1000` | Simulated bankroll |
| `TRADING_FRACTIONAL_KELLY` | `0.5` | Half-Kelly for conservative sizing |

## Testing

```bash
python -m pytest tests/ -v
```

## Tech Stack

- **Framework**: [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch)
- **Prediction**: [Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model
- **LLM**: OpenRouter (google/gemma-2-9b-it:free)
- **Data**: Apify + Binance public API
- **Risk**: Kelly Criterion with fractional sizing
- **Storage**: SQLite via aiosqlite
- **Dashboard**: FastAPI + Uvicorn
- **Logging**: structlog (structured JSON logging)

## Project Structure

```
CryptoAssing/
├── main.py                  # Entry point
├── setup_kronos.py          # Kronos model setup
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── .env.example             # Environment template
├── agents/                  # Agent implementations
│   ├── base.py              # BaseAgent + AgentContext
│   ├── market_scout.py      # Polymarket/Kalshi scanner
│   ├── data_engineer.py     # OHLCV data fetcher
│   ├── quant_agent.py       # Kronos prediction
│   ├── risk_manager.py      # Kelly criterion
│   └── evaluator.py         # Outcome evaluation + feedback
├── core/                    # Core infrastructure
│   ├── config.py            # Settings from .env
│   ├── hermes_runtime.py    # Hermes Agent SDK integration
│   ├── market_sources.py    # API clients for Poly/Kalshi
│   ├── models.py            # Data models
│   ├── orchestrator.py      # Main trading loop + feedback
│   ├── risk.py              # Kelly formula
│   └── ledger.py            # SQLite persistence
├── models/                  # ML model wrappers
│   └── kronos.py            # Kronos SDK wrapper
├── utils/                   # Utilities
│   ├── apify_client.py      # OHLCV fetching (Apify + Binance)
│   └── kronos_inference.py  # Kronos inference engine
├── api/                     # Dashboard
│   └── dashboard.py         # FastAPI REST API
├── tests/                   # Unit tests
│   └── test_core.py
└── ledger/                  # SQLite database
```

## License

MIT
