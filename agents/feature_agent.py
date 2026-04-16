"""
Feature Engineering Agent
──────────────────────────
Receives raw OHLCV DataFrames from the Data Agent and returns a dict of
enriched DataFrames plus a dict of scalar technical scores per ticker.

Also builds the model-ready price array (normalised close prices) used
by the forecasting agents.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger
from utils.indicators import build_feature_frame, compute_technical_score

logger = setup_logger("feature_agent")


class FeatureAgent:
    def __init__(self, config: dict):
        self.config = config

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
        featured_data  : {ticker -> enriched DataFrame}
        tech_scores    : {ticker -> scalar score in [0, 1]}
        price_arrays   : {ticker -> 1-D float64 numpy array of Close prices}
        """
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
                logger.debug("%s — %d rows, tech_score=%.3f", ticker, len(enriched), tech_scores[ticker])
            except Exception as exc:
                logger.error("Feature engineering failed for %s: %s", ticker, exc)

        logger.info("Feature engineering complete (%d tickers)", len(featured_data))
        return featured_data, tech_scores, price_arrays

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to *df*."""
        df = df.copy()

        # Drop rows with zero / NaN Close (bad data)
        df = df[df["Close"] > 0].dropna(subset=["Close"])

        # Ensure DatetimeIndex; strip timezone so all series are tz-naive
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()

        # Build full indicator set (see utils/indicators.py)
        df = build_feature_frame(df)

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
            "sma_20", "sma_50", "sma_200", "bb_pct", "volatility_21d",
            "momentum_21d", "return_1d", "volume_ratio",
        ]
        return {k: round(float(row[k]), 4) for k in keys if k in row and not pd.isna(row[k])}
