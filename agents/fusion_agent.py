"""
Signal Fusion Agent
────────────────────
Combines six independent signals into a single composite score [0, 1]
for each ticker × horizon combination.

Signals and default weights (configurable in config.yaml):
  • TimesFM prediction    — 28 %
  • Chronos prediction    — 20 %
  • Technical indicators  — 20 %
  • News sentiment        — 12 %
  • Fundamentals          — 12 %
  • Macro regime          —  8 %

The forecast signals are converted to a directional strength:
  score = sigmoid(predicted_pct_change / sensitivity)
  where sensitivity = 5 % (so a +10 % forecast → ~0.88 score)
"""

import math
from typing import Dict, Optional

import numpy as np

from utils.helpers import normalize_scores, setup_logger

logger = setup_logger("fusion_agent")

_SIGMOID_SENSITIVITY = 5.0   # % change that maps to ~73 % score


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _forecast_to_score(pct_change: float) -> float:
    """Map a predicted % change to a [0, 1] bullishness score."""
    return _sigmoid(pct_change / _SIGMOID_SENSITIVITY)


def _sentiment_to_score(sentiment: float) -> float:
    """Shift sentiment from [-1, +1] to [0, 1]."""
    return (sentiment + 1.0) / 2.0


class FusionAgent:
    def __init__(self, config: dict):
        w = config["fusion"]
        self.w_timesfm      = w["timesfm_weight"]
        self.w_chronos      = w["chronos_weight"]
        self.w_technical    = w["technical_weight"]
        self.w_sentiment    = w["sentiment_weight"]
        self.w_fundamentals = w.get("fundamentals_weight", 0.0)
        self.w_macro        = w.get("macro_weight", 0.0)
        self._validate_weights()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        timesfm_preds:  Dict[str, Dict[str, dict]],
        chronos_preds:  Dict[str, dict],
        tech_scores:    Dict[str, float],
        sentiment:      Dict[str, float],
        fundamentals:   Optional[Dict[str, float]] = None,
        macro:          Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns
        -------
        {
            ticker: {
                "short":  composite_score,   # float in [0, 1]
                "medium": composite_score,
                "long":   composite_score,
                "overall": composite_score,  # weighted mean of horizons
            }
        }
        """
        fundamentals = fundamentals or {}
        macro        = macro        or {}

        all_tickers = (
            set(timesfm_preds) | set(chronos_preds)
            | set(tech_scores) | set(sentiment)
        )
        logger.info("Fusing signals for %d tickers …", len(all_tickers))

        raw_scores: Dict[str, Dict[str, float]] = {}

        for ticker in all_tickers:
            scores_per_horizon: Dict[str, float] = {}

            for horizon in ["short", "medium", "long"]:
                # ── TimesFM signal ─────────────────────────────────────────
                tfm = timesfm_preds.get(ticker, {}).get(horizon, {})
                s_tfm = _forecast_to_score(tfm.get("pct_change", 0.0))

                # ── Chronos signal ─────────────────────────────────────────
                chr_ = chronos_preds.get(ticker, {}).get(horizon, {})
                s_chr = _forecast_to_score(chr_.get("pct_change", 0.0))

                # Penalise if models diverge (agreement_score in chronos output)
                agreement = chronos_preds.get(ticker, {}).get("agreement_score", 0.5)
                s_chr = s_chr * agreement + 0.5 * (1.0 - agreement)

                # ── Technical signal ───────────────────────────────────────
                s_tech = tech_scores.get(ticker, 0.5)

                # ── Sentiment signal ───────────────────────────────────────
                s_sent = _sentiment_to_score(sentiment.get(ticker, 0.0))

                # ── Fundamentals signal ────────────────────────────────────
                s_fund = fundamentals.get(ticker, 0.5)

                # ── Macro regime signal ────────────────────────────────────
                s_macro = macro.get(ticker, 0.5)

                # ── Weighted sum ───────────────────────────────────────────
                composite = (
                    self.w_timesfm      * s_tfm
                    + self.w_chronos    * s_chr
                    + self.w_technical  * s_tech
                    + self.w_sentiment  * s_sent
                    + self.w_fundamentals * s_fund
                    + self.w_macro      * s_macro
                )
                scores_per_horizon[horizon] = round(float(composite), 4)

            scores_per_horizon["overall"] = round(
                float(np.mean([scores_per_horizon[h] for h in ["short", "medium", "long"]])),
                4,
            )
            raw_scores[ticker] = scores_per_horizon

        # Per-horizon normalisation so relative rankings are sharp
        fused = self._normalise_per_horizon(raw_scores)
        logger.info("Signal fusion complete")
        return fused

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_per_horizon(
        raw: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Min-max normalise scores across tickers separately for each horizon."""
        horizons = ["short", "medium", "long", "overall"]
        normalised = {ticker: {} for ticker in raw}

        for h in horizons:
            h_scores = {t: raw[t].get(h, 0.5) for t in raw}
            normed   = normalize_scores(h_scores)
            for t, v in normed.items():
                normalised[t][h] = round(v, 4)

        return normalised

    def _validate_weights(self):
        total = (
            self.w_timesfm + self.w_chronos + self.w_technical
            + self.w_sentiment + self.w_fundamentals + self.w_macro
        )
        if abs(total - 1.0) > 1e-3:
            logger.warning("Fusion weights sum to %.3f (expected 1.0) — auto-normalising", total)
            self.w_timesfm      /= total
            self.w_chronos      /= total
            self.w_technical    /= total
            self.w_sentiment    /= total
            self.w_fundamentals /= total
            self.w_macro        /= total
