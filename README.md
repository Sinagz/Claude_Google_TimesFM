# Multi-Agent Stock Forecasting System

A production-grade Python pipeline that combines **Google TimesFM 2.5** and **Amazon Chronos Bolt** with technical analysis, FinBERT sentiment, fundamental ratios, and macro indicators to rank and forecast US & Canadian stocks across three time horizons.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                            │
│                                                                  │
│  Stage 1  DataAgent          — yfinance / Alpha Vantage OHLCV   │
│  Stage 2  FeatureAgent       — RSI, MACD, Bollinger, ATR …      │
│  Stage 3  TimesFMAgent  ★    — Google TimesFM 2.5 (200M params) │
│  Stage 4  ChronosAgent  ★    — Amazon Chronos Bolt Small        │
│  Stage 5  SentimentAgent     — FinBERT (ProsusAI) on news       │
│  Stage 5b FundamentalsAgent  — P/E, EPS growth, ROE, D/E        │
│  Stage 5c MacroAgent         — VIX, DXY, TLT, GLD, Oil, TNX    │
│  Stage 6  FusionAgent        — Weighted signal combination      │
│  Stage 7  RankingAgent       — Top-5 per horizon                │
│  Stage 8  BacktestAgent      — 1-year equity curve simulation   │
└─────────────────────────────────────────────────────────────────┘
  ★ = CRITICAL stage — pipeline aborts if model cannot load
```

## Signal Fusion Weights

| Signal | Weight |
|---|---|
| Google TimesFM 2.5 | 28 % |
| Amazon Chronos Bolt | 20 % |
| Technical indicators | 20 % |
| News sentiment (FinBERT) | 12 % |
| Fundamentals (P/E, EPS, ROE…) | 12 % |
| Macro regime (VIX, DXY, TLT…) | 8 % |

---

## Features

- **Dual forecasting models** — TimesFM 2.5 (200M parameter transformer) + Chronos Bolt (50× faster than original T5-small) for independent point + probabilistic forecasts
- **Three horizons** — 1 month (21 trading days), 6 months (126), 1 year (252)
- **65+ tickers** — US mega-cap, financials, healthcare, consumer, energy; 15 Canadian (TSX); 10 ETFs
- **GPU auto-detection** — CUDA if available, clean CPU fallback
- **Interactive HTML dashboard** — 6-tab Plotly dark-theme dashboard (rankings, price+forecast, heatmap, signal breakdown, equity curve, sentiment)
- **Rich terminal report** — colour-coded ASCII tables saved to `outputs/report.txt`
- **Windows Task Scheduler** — `schedule_setup.bat` registers a weekday 07:00 AM cron job
- **Model divergence warnings** — flags tickers where TimesFM and Chronos disagree

---

## Sample Output  *(run: 2026-04-16, 5 tickers, CPU-only, 47 s)*

```
╔════════════════════════════════════════════════════════════════════════╗
║ Multi-Agent Stock Forecasting System                                   ║
║ Run date: 2026-04-16  17:13:20  |  Tickers: 5  |  Elapsed: 47.0 s    ║
║ Models: TimesFM-2.5, Chronos-Bolt-Small                               ║
╚════════════════════════════════════════════════════════════════════════╝

TOP 5 STOCKS BY HORIZON
┌────────┬────────────────┬────────────────┬────────────────┐
│  Rank  │    1 Month     │    6 Months    │     1 Year     │
├────────┼────────────────┼────────────────┼────────────────┤
│   #1   │      NVDA      │      NVDA      │      NVDA      │
│   #2   │     GOOGL      │     GOOGL      │     GOOGL      │
│   #3   │      AAPL      │      AAPL      │      AAPL      │
│   #4   │      MSFT      │      MSFT      │      MSFT      │
│   #5   │      AMZN      │      AMZN      │      AMZN      │
└────────┴────────────────┴────────────────┴────────────────┘

── NVDA  (ranked #1 across all horizons) ──────────────────────────────
  Latest price         $198.23
  Overall score        [████████████] 1.00
  Technical score      [███████░░░░░] 0.59
  Sentiment            +0.320  (bullish)
  Model agreement      67%

  Horizon   TimesFM point   TimesFM %   Chronos point   Chronos %   Low / High
  1 Month        $197.48      -0.38%          $205.00      +3.42%   $181 – $229
  6 Months       $199.16      +0.47%          $211.00      +6.44%   $170 – $258
  1 Year         $198.63      +0.20%          $211.00      +6.44%   $170 – $258

── GOOGL ──────────────────────────────────────────────────────────────
  Latest price         $335.07
  Overall score        [██████░░░░░░] 0.54
  Sentiment            +0.155  (bullish)

  Horizon   TimesFM point   TimesFM %   Chronos point   Chronos %   Low / High
  1 Month        $338.01      +0.88%          $324.00      -3.30%   $280 – $366
  6 Months       $333.72      -0.40%          $320.00      -4.50%   $247 – $404
  1 Year         $328.82      -1.86%          $320.00      -4.50%   $247 – $404

── AAPL  ⚠ DIVERGENCE WARNING ────────────────────────────────────────
  Latest price         $264.19
  Model agreement      0%   ← TimesFM bearish, Chronos bullish

  Horizon   TimesFM point   TimesFM %   Chronos point   Chronos %   Low / High
  1 Month        $259.38      -1.82%          $270.00      +2.20%   $248 – $292
  6 Months       $251.98      -4.62%          $280.00      +5.98%   $241 – $322
  1 Year         $253.07      -4.21%          $280.00      +5.98%   $241 – $322

BACKTEST RESULTS  (last 365 calendar days)
┌────────────────────────────────┬──────────────┬──────────────┐
│ Metric                         │     Strategy │    Benchmark │
├────────────────────────────────┼──────────────┼──────────────┤
│ Cumulative return              │     +36.89%  │     +35.53%  │
│ Annualised return              │     +24.21%  │           —  │
│ Alpha (vs. benchmark)          │      +1.36%  │           —  │
│ Sharpe ratio                   │        0.98  │           —  │
│ Max drawdown                   │     -26.42%  │           —  │
│ Rebalances                     │          12  │           —  │
└────────────────────────────────┴──────────────┴──────────────┘
```

---

## Setup

### 1. Prerequisites

- Python 3.10 – 3.12
- 4 GB+ RAM (8 GB recommended for full ticker list)
- Internet access for model downloads (~800 MB on first run)

### 2. Install

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Claude_Google_TimesFM.git
cd Claude_Google_TimesFM

# Windows — one-shot setup (creates venv + installs all deps)
setup.bat

# Manual (any OS)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install plotly
```

> **TimesFM** requires Python ≥ 3.10 and must be installed from GitHub source:
> ```bash
> pip install --no-deps git+https://github.com/google-research/timesfm.git
> ```
> `setup.bat` handles this automatically.

### 3. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml and add your API keys
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

# Sanity check (data + features only, no models)
python main.py --dry-run

# Force re-fetch all data (ignore cache)
python main.py --no-cache
```

### 5. GPU Acceleration (optional)

```bash
# Auto-detects CUDA version and installs matching torch wheel
scripts\setup_gpu.bat
```

### 6. Schedule daily runs (Windows)

```bash
# Run as Administrator — registers Mon–Fri 07:00 AM Task Scheduler job
schedule_setup.bat
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/results.json` | Full structured results with all scores and forecasts |
| `outputs/report.txt` | Plain-text version of the terminal report |
| `outputs/dashboard.html` | Self-contained interactive Plotly dashboard |

### Dashboard tabs

1. **Rankings** — colour-coded top-5 table per horizon
2. **Forecast: TICKER** — 180-day price history + TimesFM & Chronos forecast lines with CI bands
3. **Score Heatmap** — all tickers × horizons composite score grid
4. **Signal Breakdown** — grouped bar chart of all 6 signals per top ticker
5. **Backtest** — equity curve vs buy-and-hold benchmark
6. **Sentiment** — FinBERT sentiment bar chart for all tickers

---

## Project Structure

```
Claude_Google_TimesFM/
├── agents/
│   ├── orchestrator.py        # Pipeline controller
│   ├── data_agent.py          # OHLCV ingestion (yfinance + Alpha Vantage)
│   ├── feature_agent.py       # Technical indicators
│   ├── timesfm_agent.py       # Google TimesFM 2.5 forecasting
│   ├── chronos_agent.py       # Amazon Chronos Bolt forecasting
│   ├── sentiment_agent.py     # FinBERT news sentiment
│   ├── fundamentals_agent.py  # P/E, EPS, ROE, D/E scoring
│   ├── macro_agent.py         # VIX, DXY, TLT, GLD, Oil, TNX regime
│   ├── fusion_agent.py        # Weighted signal combination
│   ├── ranking_agent.py       # Top-N ranking per horizon
│   └── backtest_agent.py      # Historical strategy simulation
├── utils/
│   ├── dashboard.py           # Plotly HTML dashboard generator
│   ├── reporter.py            # Rich terminal report
│   ├── helpers.py             # Config, logging, utilities
│   └── api_clients.py         # Alpha Vantage & Finnhub wrappers
├── models/
│   └── model_loader.py        # HuggingFace model availability checks
├── scripts/
│   └── setup_gpu.bat          # CUDA auto-detect + torch wheel installer
├── config.yaml.example        # Configuration template (copy → config.yaml)
├── requirements.txt
├── setup.bat                  # One-shot Windows setup
├── schedule_setup.bat         # Windows Task Scheduler registration
└── main.py                    # Entry point
```

---

## Models

| Model | Source | Size | Speed (CPU) |
|---|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | ~800 MB | ~5 s/ticker |
| Chronos Bolt Small | `amazon/chronos-bolt-small` | ~300 MB | ~0.6 s/ticker |
| FinBERT | `ProsusAI/finbert` | ~440 MB | ~2 s/ticker |

Models are downloaded automatically on first run and cached in `models/cache/`.

---

## License

MIT
