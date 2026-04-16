"""
Data Ingestion Agent
────────────────────
Fetches OHLCV history for every configured ticker.

Strategy
  1. Try yfinance (fast, free, handles both US and .TO Canadian tickers)
  2. Fall back to Alpha Vantage (rate-limited; used sparingly)

Saves one CSV per ticker under  data/<TICKER>.csv
Returns a dict  {ticker: pd.DataFrame}  for downstream agents.
"""

import os
from typing import Dict, Optional

import pandas as pd

from utils.api_clients import AlphaVantageClient, YFinanceClient
from utils.helpers import is_cache_fresh, setup_logger, get_all_tickers

logger = setup_logger("data_agent")


class DataAgent:
    def __init__(self, config: dict):
        self.config = config
        self.save_path = config["data"]["save_path"]
        self.period_days = config["data"]["period_days"]
        self.cache_hours = config["data"].get("cache_hours", 6)
        self.yf = YFinanceClient()
        self.av = AlphaVantageClient(config["alphavantage_api_key"])
        os.makedirs(self.save_path, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all tickers; return dict of DataFrames."""
        tickers = get_all_tickers(self.config)
        results: Dict[str, pd.DataFrame] = {}

        logger.info("Starting data ingestion for %d tickers", len(tickers))

        for ticker in tickers:
            df = self._load_or_fetch(ticker)
            if df is not None and not df.empty:
                results[ticker] = df
            else:
                logger.warning("No data obtained for %s — skipping", ticker)

        logger.info("Data ingestion complete: %d/%d tickers loaded", len(results), len(tickers))
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _csv_path(self, ticker: str) -> str:
        safe = ticker.replace(".", "_")
        return os.path.join(self.save_path, f"{safe}.csv")

    def _load_or_fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return cached CSV if fresh; otherwise fetch and save."""
        path = self._csv_path(ticker)

        if is_cache_fresh(path, self.cache_hours):
            logger.debug("Cache hit for %s", ticker)
            return self._read_csv(path)

        logger.info("Fetching %s …", ticker)
        df = self._fetch_yfinance(ticker)

        if df is None:
            logger.warning("yfinance failed for %s; trying Alpha Vantage …", ticker)
            df = self._fetch_alpha_vantage(ticker)

        if df is not None and not df.empty:
            df.to_csv(path)
            logger.info("Saved %s  (%d rows)", path, len(df))

        return df

    # ── yfinance ──────────────────────────────────────────────────────────────

    def _fetch_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """Pull ~2 years of daily OHLCV from yfinance."""
        # yfinance period codes: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        years = max(1, self.period_days // 365)
        period = f"{min(years, 2)}y"
        df = self.yf.get_history(ticker, period=period, interval="1d")
        if df is None:
            return None
        df = self._normalise_columns(df)
        return df

    # ── Alpha Vantage fallback ────────────────────────────────────────────────

    def _fetch_alpha_vantage(self, ticker: str) -> Optional[pd.DataFrame]:
        """Pull daily adjusted OHLCV from Alpha Vantage (rate-limited)."""
        raw = self.av.get_daily_adjusted(ticker, outputsize="full")
        if raw is None:
            return None

        key = "Time Series (Daily)"
        if key not in raw:
            logger.warning("Unexpected AV response for %s: %s", ticker, list(raw.keys()))
            return None

        ts = raw[key]
        rows = []
        for date_str, vals in ts.items():
            rows.append({
                "Date":   date_str,
                "Open":   float(vals.get("1. open", 0)),
                "High":   float(vals.get("2. high", 0)),
                "Low":    float(vals.get("3. low", 0)),
                "Close":  float(vals.get("5. adjusted close", vals.get("4. close", 0))),
                "Volume": float(vals.get("6. volume", 0)),
            })

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # keep only the requested look-back
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.period_days)
        df = df[df.index >= cutoff]
        return df

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard column names regardless of source."""
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].copy()

    @staticmethod
    def _read_csv(path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None
