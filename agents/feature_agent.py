"""
Feature Engineering Agent
──────────────────────────
Receives raw OHLCV DataFrames from the Data Agent and returns a dict of
enriched DataFrames plus a dict of scalar technical scores per ticker.

Also builds the model-ready price array (normalised close prices) used
by the forecasting agents.

Market features appended to every ticker:
  spy_return_21d  — rolling 21-day log return of SPY (market direction)
  vix_level       — VIX / 100  (fear gauge; fallback = SPY rolling vol)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger
from utils.indicators import build_feature_frame, compute_technical_score

logger = setup_logger("feature_agent")


class FeatureAgent:
    def __init__(self, config: dict):
        self.config = config
        self._market_df: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        raw_data: Dict[str, pd.DataFrame],
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], Dict[str, np.ndarray]]:
        """
        Parameters
        ----------
        raw_data : {ticker -> OHLCV DataFrame}

        Returns
        -------
        featured_data  : {ticker -> enriched DataFrame with market features}
        tech_scores    : {ticker -> scalar score in [0, 1]}
        price_arrays   : {ticker -> 1-D float64 numpy array of Close prices}
        """
        self._load_market_features()

        featured_data: Dict[str, pd.DataFrame] = {}
        tech_scores:   Dict[str, float]        = {}
        price_arrays:  Dict[str, np.ndarray]   = {}

        logger.info("Computing features for %d tickers …", len(raw_data))

        for ticker, df in raw_data.items():
            try:
                enriched = self._engineer(df)
                featured_data[ticker] = enriched
                tech_scores[ticker]   = compute_technical_score(enriched)
                price_arrays[ticker]  = self._price_array(enriched)
                logger.debug(
                    "%s — %d rows, tech_score=%.3f",
                    ticker, len(enriched), tech_scores[ticker],
                )
            except Exception as exc:
                logger.error("Feature engineering failed for %s: %s", ticker, exc)

        logger.info("Feature engineering complete (%d tickers)", len(featured_data))
        return featured_data, tech_scores, price_arrays

    # ── Market features ───────────────────────────────────────────────────────

    def _load_market_features(self) -> None:
        """Fetch SPY + VIX once per run; store aligned DataFrame."""
        if self._market_df is not None:
            return
        try:
            import yfinance as yf

            spy_raw = yf.download(
                "SPY", period="2y", interval="1d",
                progress=False, auto_adjust=True,
            )
            if spy_raw is None or spy_raw.empty:
                raise ValueError("SPY download returned empty")

            # yfinance multi-level columns → flatten
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)

            spy_close = spy_raw["Close"].squeeze()
            mkt = pd.DataFrame(index=spy_close.index)
            mkt["spy_return_21d"] = np.log(spy_close / spy_close.shift(21))

            # Try real VIX
            try:
                vix_raw = yf.download(
                    "^VIX", period="2y", interval="1d",
                    progress=False, auto_adjust=True,
                )
                if vix_raw is not None and not vix_raw.empty:
                    if isinstance(vix_raw.columns, pd.MultiIndex):
                        vix_raw.columns = vix_raw.columns.get_level_values(0)
                    vix_close = vix_raw["Close"].squeeze()
                    mkt["vix_level"] = vix_close.reindex(mkt.index, method="ffill") / 100.0
                else:
                    raise ValueError("VIX empty")
            except Exception:
                # Fallback: SPY 21-day rolling annualised volatility
                log_ret = np.log(spy_close / spy_close.shift(1))
                mkt["vix_level"] = log_ret.rolling(21).std() * np.sqrt(252)

            # Normalise index to tz-naive date
            if mkt.index.tz is not None:
                mkt.index = mkt.index.tz_localize(None)
            mkt.index = mkt.index.normalize()

            self._market_df = mkt.dropna(how="all")
            logger.info(
                "Market features ready: SPY + VIX, %d rows", len(self._market_df)
            )
        except Exception as exc:
            logger.warning(
                "Market feature fetch failed (%s) — spy_return_21d/vix_level will be 0", exc
            )
            self._market_df = pd.DataFrame(columns=["spy_return_21d", "vix_level"])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators and market features to *df*."""
        df = df.copy()

        # Drop rows with zero / NaN Close (bad data)
        df = df[df["Close"] > 0].dropna(subset=["Close"])

        # Ensure DatetimeIndex; strip timezone so all series are tz-naive
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        df.index = df.index.normalize()

        # Build full indicator set (see utils/indicators.py)
        df = build_feature_frame(df)

        # Merge market features by date
        if self._market_df is not None and not self._market_df.empty:
            mkt = self._market_df.reindex(df.index, method="ffill")
            for col in ("spy_return_21d", "vix_level"):
                if col in mkt.columns:
                    df[col] = mkt[col].values

        return df

    @staticmethod
    def _price_array(df: pd.DataFrame) -> np.ndarray:
        """Return a 1-D array of Close prices, NaN-dropped, as float64."""
        arr = df["Close"].dropna().values.astype(np.float64)
        return arr

    # ── Utility: summarise latest values ─────────────────────────────────────

    @staticmethod
    def latest_indicators(df: pd.DataFrame) -> dict:
        """Return a dict of the most recent indicator values for reporting."""
        row = df.dropna(how="all").iloc[-1]
        keys = [
            "Close", "rsi_14", "macd", "macd_signal", "macd_hist",
            "sma_20", "sma_50", "sma_200", "bb_pct",
            "volatility_5d", "volatility_21d", "volatility_63d",
            "momentum_21d", "return_1d", "volume_ratio",
            "drawdown_21d", "trend_strength",
            "spy_return_21d", "vix_level",
        ]
        return {k: round(float(row[k]), 4) for k in keys if k in row and not pd.isna(row[k])}
