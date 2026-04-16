"""
Macro Context Agent  (E)
─────────────────────────
Fetches macro-economic indicators from yfinance and computes a single
market-regime score in [0, 1]:

  1.0 = risk-on (bullish macro environment)
  0.5 = neutral
  0.0 = risk-off (bearish macro environment)

Indicators tracked
  ^VIX  — CBOE Volatility Index       low=bullish, high=bearish
  UUP   — USD Index ETF               rising=mixed, falling=bullish for equities
  TLT   — 20+ Year Treasury Bond      rising=risk-off / falling yields risk-on
  GLD   — Gold ETF                    rising=risk-off safe-haven demand
  USO   — Oil ETF                     rising=inflation risk
  ^TNX  — 10-Year Treasury yield      rising quickly = tightening = bearish

The same macro_score is returned for every ticker (uniform market-wide signal).
Sector-specific adjustments can be added in a future iteration.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("macro_agent")

_MACRO_TICKERS = {
    "vix":   "^VIX",
    "usd":   "UUP",
    "bonds": "TLT",
    "gold":  "GLD",
    "oil":   "USO",
    "yield": "^TNX",
}


class MacroAgent:
    def __init__(self, config: dict):
        self.lookback = config["macro"].get("lookback_days", 30)
        self.vix_low  = config["macro"].get("vix_low",  15)
        self.vix_high = config["macro"].get("vix_high", 30)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tickers: List[str]) -> Dict[str, float]:
        """
        Returns {ticker -> macro_score} — same value for all tickers,
        representing the current market-wide macro regime.
        """
        logger.info("Fetching macro indicators …")
        score = self._compute_regime_score()
        logger.info("Macro regime score: %.3f  (%s)", score, self._label(score))
        return {t: round(score, 4) for t in tickers}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_regime_score(self) -> float:
        signals = []

        for name, yf_ticker in _MACRO_TICKERS.items():
            try:
                prices = self._fetch(yf_ticker)
                if prices is None or len(prices) < 5:
                    continue
                sig = self._score_indicator(name, prices)
                if sig is not None:
                    signals.append(sig)
                    logger.debug("  %-8s → %.3f", name, sig)
            except Exception as exc:
                logger.debug("Macro fetch failed for %s: %s", yf_ticker, exc)

        return float(np.mean(signals)) if signals else 0.5

    def _fetch(self, ticker: str):
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=f"{self.lookback + 10}d", interval="1d")
            if df.empty:
                return None
            series = df["Close"].dropna()
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            return series.tail(self.lookback)
        except Exception as exc:
            logger.debug("yfinance macro failed for %s: %s", ticker, exc)
            return None

    def _score_indicator(self, name: str, prices: pd.Series) -> float:
        current = float(prices.iloc[-1])
        lookback_mean = float(prices.mean())
        trend = (current - lookback_mean) / lookback_mean  # + = rising

        if name == "vix":
            # Low VIX → bullish; high VIX → bearish
            if current <= self.vix_low:
                level_score = 1.0
            elif current >= self.vix_high:
                level_score = 0.0
            else:
                level_score = 1.0 - (current - self.vix_low) / (self.vix_high - self.vix_low)
            # Falling VIX is also bullish
            trend_score = 0.5 - trend * 2.0   # rising VIX → lower score
            return max(0.0, min(1.0, 0.7 * level_score + 0.3 * trend_score))

        elif name == "usd":
            # Rising USD = risk-off (slightly bearish for equities overall)
            return max(0.0, min(1.0, 0.5 - trend * 1.5))

        elif name == "bonds":
            # Rising TLT = falling yields = risk-on / bullish
            return max(0.0, min(1.0, 0.5 + trend * 2.0))

        elif name == "gold":
            # Rising gold = safe-haven demand = risk-off
            return max(0.0, min(1.0, 0.5 - trend * 1.5))

        elif name == "oil":
            # Moderate oil rise is neutral; spike = inflation risk = bearish
            return max(0.0, min(1.0, 0.5 - trend * 1.0))

        elif name == "yield":
            # Rapidly rising 10Y yield = tightening = bearish
            return max(0.0, min(1.0, 0.5 - trend * 2.0))

        return 0.5

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.65:
            return "RISK-ON (bullish)"
        elif score >= 0.45:
            return "NEUTRAL"
        return "RISK-OFF (bearish)"
