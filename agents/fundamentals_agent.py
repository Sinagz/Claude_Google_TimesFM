"""
Fundamentals Agent  (D)
────────────────────────
Fetches key financial ratios per ticker and returns a composite
fundamental score in [0, 1].

Data sources (in order, first success wins per ticker):
  1. yfinance  .info  dict  — free, comprehensive
  2. Finnhub   basic financials endpoint — free tier, 60 req/min

Signals scored and normalised to [0, 1]:
  pe_score      : lower P/E is better (value); P/E > max_pe → 0
  eps_growth    : higher TTM EPS growth is better
  rev_growth    : higher TTM revenue growth is better
  debt_eq       : lower Debt/Equity is better
  roe           : higher Return on Equity is better

Final fundamental_score = equal-weight mean of available signals.
"""

from typing import Dict, List, Optional

import numpy as np

from utils.api_clients import FinnhubClient, YFinanceClient
from utils.helpers import setup_logger

logger = setup_logger("fundamentals_agent")


class FundamentalsAgent:
    def __init__(self, config: dict):
        self.fh_client  = FinnhubClient(config["finnhub_api_key"])
        self.yf_client  = YFinanceClient()
        self.max_pe     = config["fundamentals"].get("max_pe", 60)
        self.min_eps_g  = config["fundamentals"].get("min_eps_growth", -0.5)
        self.max_eps_g  = config["fundamentals"].get("max_eps_growth",  0.5)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tickers: List[str]) -> Dict[str, float]:
        """
        Returns {ticker -> fundamental_score} in [0, 1].
        Missing data → 0.5 (neutral).
        """
        scores: Dict[str, float] = {}
        logger.info("Fetching fundamentals for %d tickers …", len(tickers))

        for ticker in tickers:
            try:
                raw = self._fetch(ticker)
                scores[ticker] = self._score(raw) if raw else 0.5
            except Exception as exc:
                logger.warning("Fundamentals failed for %s: %s", ticker, exc)
                scores[ticker] = 0.5

        loaded = sum(1 for v in scores.values() if v != 0.5)
        logger.info("Fundamentals complete — %d/%d tickers scored", loaded, len(tickers))
        return scores

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch(self, ticker: str) -> Optional[dict]:
        """Try yfinance first, fall back to Finnhub."""
        data = self._from_yfinance(ticker)
        if data:
            return data
        return self._from_finnhub(ticker)

    def _from_yfinance(self, ticker: str) -> Optional[dict]:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if not info or info.get("regularMarketPrice") is None:
                return None
            return {
                "pe":         info.get("trailingPE"),
                "pb":         info.get("priceToBook"),
                "eps_growth": info.get("earningsGrowth"),        # TTM
                "rev_growth": info.get("revenueGrowth"),         # TTM
                "debt_eq":    info.get("debtToEquity"),          # as reported
                "roe":        info.get("returnOnEquity"),
            }
        except Exception as exc:
            logger.debug("yfinance info failed for %s: %s", ticker, exc)
            return None

    def _from_finnhub(self, ticker: str) -> Optional[dict]:
        try:
            raw = self.fh_client.get_basic_financials(ticker)
            if not raw:
                return None
            m = raw.get("metric", {})
            return {
                "pe":         m.get("peAnnual") or m.get("peTTM"),
                "pb":         m.get("pbAnnual"),
                "eps_growth": m.get("epsGrowth3Y"),
                "rev_growth": m.get("revenueGrowthTTMYoy"),
                "debt_eq":    m.get("totalDebt/totalEquityAnnual"),
                "roe":        m.get("roeTTM"),
            }
        except Exception as exc:
            logger.debug("Finnhub fundamentals failed for %s: %s", ticker, exc)
            return None

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, raw: dict) -> float:
        signals = []

        # P/E: 0 (overvalued) → 1 (undervalued); P/E ≤ 0 or > max_pe → 0
        pe = raw.get("pe")
        if pe and 0 < pe <= self.max_pe:
            signals.append(1.0 - pe / self.max_pe)
        elif pe and pe > self.max_pe:
            signals.append(0.0)

        # EPS growth: clamp to [min_eps_g, max_eps_g], scale to [0, 1]
        eg = raw.get("eps_growth")
        if eg is not None:
            rng = self.max_eps_g - self.min_eps_g
            signals.append(max(0.0, min(1.0, (eg - self.min_eps_g) / rng)))

        # Revenue growth: same clamp window
        rg = raw.get("rev_growth")
        if rg is not None:
            rng = self.max_eps_g - self.min_eps_g
            signals.append(max(0.0, min(1.0, (rg - self.min_eps_g) / rng)))

        # Debt/Equity: 0 (D/E=200) → 1 (D/E=0); cap at 200
        de = raw.get("debt_eq")
        if de is not None and de >= 0:
            signals.append(max(0.0, 1.0 - de / 200.0))

        # ROE: 0 (≤0%) → 1 (≥30%)
        roe = raw.get("roe")
        if roe is not None:
            signals.append(max(0.0, min(1.0, roe / 0.30)))

        return float(np.mean(signals)) if signals else 0.5
