# Multi-Agent Quant Trading Platform

A production-grade Python platform combining **Google TimesFM 2.5**, **Amazon Chronos Bolt**, a **PPO Reinforcement Learning trading agent**, **crash detection**, **IBKR auto-execution**, and a full suite of fundamental, technical, sentiment, and macro signals to rank, forecast, and autonomously trade US & Canadian equities across three time horizons.

---

## Architecture — 14-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                │
│                                                                          │
│  Stage 0   UniverseAgent        — Pre-screening: price, volume, sector  │
│  Stage 1   DataAgent            — yfinance / Alpha Vantage OHLCV        │
│  Stage 2   FeatureAgent         — RSI, MACD, Bollinger, ATR, momentum … │
│  Stage 3   TimesFMAgent    ★    — Google TimesFM 2.5 (200 M params)     │
│  Stage 4   ChronosAgent    ★    — Amazon Chronos Bolt Small             │
│  Stage 5   SentimentAgent       — VADER + NewsAPI sentiment             │
│  Stage 5b  FundamentalsAgent    — P/E, EPS growth, ROE, D/E scoring     │
│  Stage 5c  MacroAgent           — VIX, DXY, TLT, GLD, Oil, TNX regime  │
│  Stage 6   CrashDetectionAgent  — 5-signal crash probability [0, 1]     │
│  Stage 7   MLAgent              — XGBoost meta-model training+inference │
│  Stage 8   FusionAgent          — Weighted ensemble signal combination  │
│  Stage 8b  ClusteringAgent      — KMeans diversification clustering     │
│  Stage 8c  RiskEngine           — Alpha score, mispricing, regime score │
│  Stage 9   RLTradingAgent       — PPO portfolio allocation (PyTorch)    │
│  Stage 10  RankingAgent         — Growth / Value / Defensive portfolios │
│  Stage 11  IBKRExecutionAgent   — Paper/live order routing (optional)   │
│  Stage 12  BacktestAgent        — Regime-aware equity curve simulation  │
└──────────────────────────────────────────────────────────────────────────┘
  ★ = CRITICAL stage — pipeline aborts if model cannot load
```

---

## Key Components

### Crash Detection Agent
Five composite signals → logistic sharpening → crash probability in [0, 1]:

| Signal | Weight |
|---|---|
| VIX z-score | 30 % |
| SPY 21-day momentum | 25 % |
| Cross-asset correlation spike | 20 % |
| Market breadth (% stocks above 50-day MA) | 15 % |
| Vol-of-vol | 10 % |

Regime labels: **bull** (<0.25) · **neutral** (0.25–0.50) · **high_volatility** (0.50–0.75) · **crash** (≥0.75)

### RL Trading Agent (PPO)
- **Actor-Critic** shared MLP trunk → LayerNorm → ReLU ×2
- **Dirichlet actor head** — softplus concentrations → portfolio weights summing to 1
- **Reward**: portfolio return − λ_drawdown × max-drawdown − λ_vol × daily-vol
- **GAE** advantage estimation (γ=0.99, λ_GAE=0.95), PPO clipped surrogate (ε=0.2)
- Falls back to equal-weight scaled by crash probability when untrained

### IBKR Execution Agent
- Disabled and paper-mode by default (`enabled: false`, `dry_run: true`)
- **Kill-switch**: no orders placed when `crash_probability ≥ 0.65`
- Blends RL portfolio weights with ranking-agent bonus
- Enforces `max_position_pct` (15 %) and `max_total_exposure` (95 %)
- Supports `LIMIT` and `MARKET` order types via `ib_insync`

### Signal Fusion Weights

| Signal group | Weight |
|---|---|
| Forecast ensemble (TimesFM 40 % + Chronos 30 % + ML 30 %) | 50 % |
| Technical indicators | 18 % |
| News sentiment | 12 % |
| Fundamentals (P/E, EPS, ROE…) | 12 % |
| Macro regime (VIX, DXY, TLT…) | 8 % |

### Portfolio Strategies
Three simultaneous portfolios ranked from the full universe:
- **Growth** — highest forecast-ensemble + momentum composite
- **Value** — strong fundamentals + sentiment overlay
- **Defensive** — low volatility + macro tailwind

Final diversified **Top-10** drawn across strategies using KMeans clustering to cap sector concentration.

---

## Features

- **Dual SOTA forecasters** — TimesFM 2.5 (200 M transformer) + Chronos Bolt (50× faster than T5-small)
- **Three horizons** — 1 month (21 days), 6 months (126), 1 year (252 trading days)
- **75+ tickers** — US mega-cap, financials, healthcare, consumer, energy; 15 Canadian (TSX); 10 ETFs
- **PPO RL agent** — learned portfolio allocation policy trained on walk-forward historical data
- **Regime-aware backtest** — transaction costs (10 bps), slippage (5 bps), Sortino, CAGR, drawdown, per-regime P&L
- **Crash detection** — 5-signal real-time market stress monitor
- **IBKR auto-execution** — paper and live order routing with kill-switch safety
- **20-tab interactive dashboard** — Plotly dark-theme HTML (rankings, forecasts, RL portfolio, crash gauge, sector mix, risk metrics, regime performance…)
- **Rich terminal report** — colour-coded tables saved to `outputs/report.txt`
- **GPU auto-detection** — CUDA if available, clean CPU fallback
- **Windows Task Scheduler** — `schedule_setup.bat` registers a weekday 07:00 AM job
- **Dashboard from JSON** — regenerate the dashboard at any time without re-running the pipeline

---

## Setup

### 1. Prerequisites

- Python 3.10 – 3.12
- 8 GB+ RAM recommended for full ticker list
- Internet access for model downloads (~1.5 GB on first run)

### 2. Install

```bash
git clone https://github.com/Sinagz/Claude_Google_TimesFM.git
cd Claude_Google_TimesFM

# Windows — one-shot setup (creates venv + installs all deps)
setup.bat

# Manual (any OS)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **TimesFM** must be installed from GitHub (PyPI build is JAX-only and incompatible with Python 3.12):
> ```bash
> pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
> ```
> `setup.bat` handles this automatically.

### 3. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml — add API keys and tune parameters
```

| Key | Free tier | Link |
|---|---|---|
| `alphavantage_api_key` | 25 req/day | https://www.alphavantage.co/support/#api-key |
| `finnhub_api_key` | 60 req/min | https://finnhub.io/register |
| `newsapi_org_key` | 100 req/day | https://newsapi.org/register |
| `newsapi_ai_key` | optional | https://www.newsapi.ai/ |

### 4. Run

```bash
# Full pipeline (all configured tickers)
python main.py

# Quick test with specific tickers
python main.py --tickers AAPL MSFT NVDA

# Regenerate dashboard from last results without re-running pipeline
python -m utils.dashboard --json outputs/results.json

# Force re-fetch all data (ignore cache)
python main.py --no-cache
```

### 5. GPU Acceleration (optional)

```bash
scripts\setup_gpu.bat    # auto-detects CUDA version, installs matching torch wheel
```

### 6. IBKR Paper Trading (optional)

1. Install [TWS or IB Gateway](https://www.interactivebrokers.com/en/trading/tws.php) and start it on `localhost:7497`
2. Set `ibkr.enabled: true` in `config.yaml` (keep `live_trading: false` and `dry_run: true`)
3. Run `python main.py` — orders will be logged but not submitted until `dry_run: false`

### 7. Schedule daily runs (Windows)

```bash
# Run as Administrator — registers Mon–Fri 07:00 AM Task Scheduler job
schedule_setup.bat
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/results.json` | Full structured results — scores, forecasts, RL weights, crash risk, trades |
| `outputs/report.txt` | Plain-text terminal report |
| `outputs/dashboard.html` | Self-contained 20-tab interactive Plotly dashboard |
| `models/cache/rl_model.pt` | Saved PPO model weights (reused on subsequent runs) |

### Dashboard tabs (20 total)

| # | Tab | Content |
|---|---|---|
| 1 | Overview | Crash probability gauge + regime badge + recommendation |
| 2 | Top 10 | Diversified top-10 table with scores and expected returns |
| 3 | Strategies | Growth / Value / Defensive portfolio tables |
| 4 | Sector Mix | Donut chart of sector distribution in top-10 |
| 5 | RL Portfolio | Bar chart of RL-allocated weights ($ equivalent) |
| 6 | RL Allocation | Pie chart of RL portfolio + cash weight |
| 7 | Crash Risk | Signal-by-signal crash indicator bars |
| 8–N | Forecast: TICKER | 180-day price history + TimesFM & Chronos lines with CI bands |
| — | Score Heatmap | All tickers × horizons composite score grid |
| — | Signal Breakdown | Grouped bar of all 6 signals per top ticker |
| — | Risk Metrics | 2×2 subplot: alpha score, mispricing, regime score, volatility |
| — | Backtest | Equity curve vs buy-and-hold benchmark |
| — | Regime Perf | Cumulative return per market regime |
| — | Sentiment | VADER sentiment scores for all tickers |

---

## Project Structure

```
Claude_Google_TimesFM/
├── agents/
│   ├── orchestrator.py           # 14-stage pipeline controller
│   ├── data_agent.py             # OHLCV ingestion (yfinance + Alpha Vantage)
│   ├── feature_agent.py          # Technical indicators
│   ├── universe_agent.py         # Pre-screening (price, volume, sector cap)
│   ├── timesfm_agent.py          # Google TimesFM 2.5 forecasting
│   ├── chronos_agent.py          # Amazon Chronos Bolt forecasting
│   ├── sentiment_agent.py        # VADER + NewsAPI sentiment
│   ├── fundamentals_agent.py     # P/E, EPS, ROE, D/E scoring
│   ├── macro_agent.py            # VIX, DXY, TLT, GLD, Oil, TNX regime
│   ├── crash_detection_agent.py  # 5-signal crash probability
│   ├── fusion_agent.py           # Weighted signal combination
│   ├── clustering_agent.py       # KMeans diversification
│   ├── risk_engine.py            # Alpha score, mispricing, regime
│   ├── rl_trading_agent.py       # PPO portfolio allocation (PyTorch)
│   ├── ranking_agent.py          # Growth / Value / Defensive portfolios
│   ├── ibkr_execution_agent.py   # IBKR paper/live order routing
│   └── backtest_agent.py         # Regime-aware historical simulation
├── utils/
│   ├── dashboard.py              # 20-tab Plotly HTML dashboard
│   ├── reporter.py               # Rich terminal report
│   ├── helpers.py                # Config, logging, utilities
│   └── api_clients.py            # Alpha Vantage & Finnhub wrappers
├── models/
│   ├── model_loader.py           # HuggingFace model availability checks
│   └── cache/                    # Downloaded model weights + rl_model.pt
├── scripts/
│   └── setup_gpu.bat             # CUDA auto-detect + torch wheel installer
├── config.yaml.example           # Configuration template (copy → config.yaml)
├── requirements.txt
├── setup.bat                     # One-shot Windows setup
├── schedule_setup.bat            # Windows Task Scheduler registration
└── main.py                       # Entry point
```

---

## Models

| Model | HuggingFace ID | Size | Speed (CPU) |
|---|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | ~800 MB | ~5 s/ticker |
| Chronos Bolt Small | `amazon/chronos-bolt-small` | ~300 MB | ~0.6 s/ticker |
| FinBERT / VADER | `ProsusAI/finbert` / built-in | ~440 MB | ~2 s/ticker |
| PPO RL Agent | trained locally | ~2 MB | ~0.1 s/step |

Models are downloaded automatically on first run and cached in `models/cache/`.

---

## Configuration Reference

Key sections in `config.yaml`:

```yaml
rl_agent:
  training_enabled: true
  n_assets: 10          # top-N stocks the RL agent manages
  n_episodes: 8         # training episodes
  episode_len: 126      # 6 months per episode
  lambda_drawdown: 0.10 # drawdown penalty weight
  lambda_vol: 0.05      # volatility penalty weight

crash_detection:
  vix_high: 30.0        # VIX above this = stress
  vix_extreme: 40.0     # VIX above this = panic

ibkr:
  enabled: false        # master switch
  live_trading: false   # MUST remain false for paper trading
  dry_run: true         # log-only, no actual order submissions
  kill_switch_threshold: 0.65  # crash_prob above this → no trades

backtest:
  transaction_cost_bps: 10
  slippage_bps: 5
```

See `config.yaml.example` for the full annotated reference.

---

## Safety

- IBKR live trading is **disabled by default** and requires three explicit opt-ins (`enabled`, `live_trading`, `dry_run`)
- The RL agent checks crash probability before proposing any allocation
- The IBKR kill-switch hard-stops execution at `crash_probability ≥ 0.65`
- No API keys are committed — `config.yaml` is in `.gitignore`

---

## License

MIT
