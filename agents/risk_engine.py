"""
Risk Engine
────────────
Computes risk-adjusted alpha scores, mispricing vs. sector/market,
and detects the current market regime.

Alpha Score
  alpha_score = (expected_return * confidence) / (volatility * liquidity_penalty)
  Where:
    expected_return = weighted average of ML, TimesFM, Chronos predicted returns
    confidence      = ML short-horizon confidence (or 0.5 fallback)
    volatility      = annualised 21-day rolling vol from featured_data
    liquidity_penalty = 1.0 for normal volume, up to 2.0 for low volume

Mispricing Score
  mispricing = expected_return - sector_avg_return - market_return_proxy

Regime Detection
  Uses SPY 21-day return + VIX level from featured_data:
    bull         : spy_return_21d > 0.03  and  vix < 0.20
    bear         : spy_return_21d < -0.03 or   vix > 0.30
    high_volatility: vix > 0.25
    neutral      : otherwise

Output per ticker
  {
    "alpha_score":    float,
    "mispricing":     float,
    "sharpe_proxy":   float,
    "drawdown_risk":  float,   # magnitude of recent drawdown
    "regime":         str,     # bull / bear / high_volatility / neutral
  }
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("risk_engine")


class RiskEngine:
    def __init__(self, config: dict):
        risk = config.get("risk", {})
        self.min_vol          = float(risk.get("min_vol", 0.05))
        self.liquidity_threshold = float(risk.get("liquidity_vol_threshold", 500_000))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        featured_data:  Dict[str, pd.DataFrame],
        ml_predictions: Dict[str, Dict[str, dict]],
        timesfm_preds:  Dict[str, Dict[str, dict]],
        chronos_preds:  Dict[str, Dict[str, dict]],
    ) -> dict:
        """
        Returns
        -------
        {
            "scores":  {ticker: {alpha_score, mispricing, sharpe_proxy, drawdown_risk}},
            "regime":  str,
        }
        """
        regime = self._detect_regime(featured_data)
        logger.info("Market regime: %s", regime)

        sector_avg = self._sector_avg_return(ml_predictions, timesfm_preds, chronos_preds)
        market_ret = self._market_return_proxy(featured_data)

        scores: dict = {}
        for ticker, df in featured_data.items():
            if df is None or df.empty:
                continue
            scores[ticker] = self._score_ticker(
                ticker, df,
                ml_predictions.get(ticker, {}),
                timesfm_preds.get(ticker, {}),
                chronos_preds.get(ticker, {}),
                sector_avg,
                market_ret,
            )

        logger.info("Risk engine scored %d tickers", len(scores))
        return {"scores": scores, "regime": regime}

    # ── Regime detection ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_regime(featured_data: Dict[str, pd.DataFrame]) -> str:
        spy_ret = None
        vix_lvl = None

        # Pull SPY market features from any ticker's featured_data
        for df in featured_data.values():
            if df is None or df.empty:
                continue
            if "spy_return_21d" in df.columns:
                val = df["spy_return_21d"].dropna()
                if not val.empty:
                    spy_ret = float(val.iloc[-1])
            if "vix_level" in df.columns:
                val = df["vix_level"].dropna()
                if not val.empty:
                    vix_lvl = float(val.iloc[-1])
            if spy_ret is not None and vix_lvl is not None:
                break

        if spy_ret is None:
            spy_ret = 0.0
        if vix_lvl is None:
            vix_lvl = 0.20  # neutral assumption

        if vix_lvl > 0.30:
            return "high_volatility"
        if spy_ret < -0.03 or vix_lvl > 0.25:
            return "bear"
        if spy_ret > 0.03 and vix_lvl < 0.20:
            return "bull"
        return "neutral"

    # ── Sector average return ─────────────────────────────────────────────────

    @staticmethod
    def _sector_avg_return(
        ml_preds:      Dict[str, Dict],
        timesfm_preds: Dict[str, Dict],
        chronos_preds: Dict[str, Dict],
    ) -> float:
        returns = []
        for ticker in ml_preds:
            short = ml_preds[ticker].get("short", {})
            r = short.get("predicted_return", None)
            if r is not None:
                returns.append(float(r))
        if not returns:
            for ticker in timesfm_preds:
                p = timesfm_preds[ticker].get("short", {})
                pct = p.get("pct_change", None)
                if pct is not None:
                    returns.append(float(pct) / 100.0)
        return float(np.mean(returns)) if returns else 0.0

    # ── Market return proxy ───────────────────────────────────────────────────

    @staticmethod
    def _market_return_proxy(featured_data: Dict[str, pd.DataFrame]) -> float:
        for df in featured_data.values():
            if df is None or df.empty:
                continue
            if "spy_return_21d" in df.columns:
                val = df["spy_return_21d"].dropna()
                if not val.empty:
                    return float(val.iloc[-1])
        return 0.0

    # ── Per-ticker scoring ────────────────────────────────────────────────────

    def _score_ticker(
        self,
        ticker:        str,
        df:            pd.DataFrame,
        ml_pred:       dict,
        timesfm_pred:  dict,
        chronos_pred:  dict,
        sector_avg:    float,
        market_ret:    float,
    ) -> dict:
        # Expected return: weighted ensemble
        ml_ret  = float(ml_pred.get("short", {}).get("predicted_return", 0.0) or 0.0)
        tfm_pct = float(timesfm_pred.get("short", {}).get("pct_change", 0.0) or 0.0) / 100.0
        chr_pct = float(chronos_pred.get("short", {}).get("pct_change", 0.0) or 0.0) / 100.0

        has_ml  = bool(ml_pred and "short" in ml_pred)
        if has_ml:
            expected_ret = 0.40 * ml_ret + 0.30 * tfm_pct + 0.30 * chr_pct
        else:
            expected_ret = 0.50 * tfm_pct + 0.50 * chr_pct

        # Confidence
        confidence = float(ml_pred.get("short", {}).get("confidence", 0.5) or 0.5)

        # Volatility
        vol = self.min_vol
        if "volatility_21d" in df.columns:
            v = df["volatility_21d"].dropna()
            if not v.empty:
                vol = max(self.min_vol, float(v.iloc[-1]))

        # Liquidity factor (penalise low-volume tickers)
        liquidity_factor = 1.0
        if "Volume" in df.columns:
            avg_vol = df["Volume"].tail(21).mean()
            if avg_vol < self.liquidity_threshold:
                liquidity_factor = max(0.5, avg_vol / self.liquidity_threshold)

        # Alpha score
        alpha_score = (expected_ret * confidence * liquidity_factor) / vol

        # Mispricing vs sector + market
        mispricing = expected_ret - sector_avg - market_ret

        # Sharpe proxy: expected_return / volatility (annualised scale)
        sharpe_proxy = expected_ret / vol if vol > 0 else 0.0

        # Drawdown risk
        drawdown_risk = 0.0
        if "drawdown_21d" in df.columns:
            dd = df["drawdown_21d"].dropna()
            if not dd.empty:
                drawdown_risk = abs(float(dd.iloc[-1]))

        return {
            "alpha_score":   round(float(alpha_score),   4),
            "mispricing":    round(float(mispricing),    4),
            "sharpe_proxy":  round(float(sharpe_proxy),  4),
            "drawdown_risk": round(float(drawdown_risk), 4),
            "expected_return": round(float(expected_ret), 4),
            "confidence":    round(float(confidence),    4),
            "volatility":    round(float(vol),           4),
        }
