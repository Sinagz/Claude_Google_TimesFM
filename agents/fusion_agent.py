"""
Signal Fusion Agent
────────────────────
Combines signals into a composite score [0, 1] per ticker × horizon.

Architecture
  Step 1 — Forecast ensemble (three independent predictive models):
              forecast_score = 0.40 × TimesFM + 0.30 × Chronos + 0.30 × ML
  Step 2 — Overall composite weighted across all signal sources:
              composite = w_forecast × forecast_score
                        + w_technical × tech_score
                        + w_sentiment × sentiment_score
                        + w_fundamentals × fundamentals_score
                        + w_macro × macro_score

Default weights (all configurable in config.yaml under `fusion:`):
  Forecast ensemble weight  : 50 %
    └─ TimesFM sub-weight   : 40 %
    └─ Chronos sub-weight   : 30 %
    └─ ML meta-model weight : 30 %
  Technical                 : 18 %
  Sentiment                 : 12 %
  Fundamentals              : 12 %
  Macro                     :  8 %
"""

import math
from typing import Dict, Optional

import numpy as np

from utils.helpers import normalize_scores, setup_logger

logger = setup_logger("fusion_agent")

_SIGMOID_SENSITIVITY = 5.0   # % → ~73 % score at +5 %


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _forecast_to_score(pct_change: float) -> float:
    """Map predicted % change to [0, 1] bullishness score via sigmoid."""
    return _sigmoid(pct_change / _SIGMOID_SENSITIVITY)


def _sentiment_to_score(sentiment: float) -> float:
    """Shift sentiment from [-1, +1] to [0, 1]."""
    return (sentiment + 1.0) / 2.0


class FusionAgent:
    def __init__(self, config: dict):
        w = config["fusion"]

        # ── Forecast sub-ensemble weights (sum to 1.0) ─────────────────────
        self.ens_w_tfm = w.get("ensemble_timesfm_weight", 0.40)
        self.ens_w_chr = w.get("ensemble_chronos_weight",  0.30)
        self.ens_w_ml  = w.get("ensemble_ml_weight",       0.30)

        # ── Overall signal weights (sum to 1.0) ────────────────────────────
        # Backward-compatible: if old keys exist, derive forecast_ensemble_weight
        _old_forecast = (
            w.get("timesfm_weight", 0.28) + w.get("chronos_weight", 0.20)
        )
        self.w_forecast     = w.get("forecast_ensemble_weight", _old_forecast)
        self.w_technical    = w.get("technical_weight",    0.18)
        self.w_sentiment    = w.get("sentiment_weight",    0.12)
        self.w_fundamentals = w.get("fundamentals_weight", 0.12)
        self.w_macro        = w.get("macro_weight",        0.08)

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
        ml_predictions: Optional[Dict[str, Dict[str, dict]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns
        -------
        {
            ticker: {
                "short":   composite_score,   # float in [0, 1]
                "medium":  composite_score,
                "long":    composite_score,
                "overall": composite_score,   # mean of horizons
            }
        }
        """
        fundamentals   = fundamentals   or {}
        macro          = macro          or {}
        ml_predictions = ml_predictions or {}

        all_tickers = (
            set(timesfm_preds) | set(chronos_preds)
            | set(tech_scores) | set(sentiment)
        )
        logger.info("Fusing signals for %d tickers …", len(all_tickers))

        raw_scores: Dict[str, Dict[str, float]] = {}

        for ticker in all_tickers:
            scores_per_horizon: Dict[str, float] = {}

            for horizon in ("short", "medium", "long"):

                # ── TimesFM signal ─────────────────────────────────────────
                tfm    = timesfm_preds.get(ticker, {}).get(horizon, {})
                s_tfm  = _forecast_to_score(tfm.get("pct_change", 0.0))

                # ── Chronos signal (penalised by disagreement) ─────────────
                chr_        = chronos_preds.get(ticker, {}).get(horizon, {})
                s_chr_raw   = _forecast_to_score(chr_.get("pct_change", 0.0))
                agreement   = chronos_preds.get(ticker, {}).get("agreement_score", 0.5)
                s_chr       = s_chr_raw * agreement + 0.5 * (1.0 - agreement)

                # ── ML meta-model signal ───────────────────────────────────
                ml   = ml_predictions.get(ticker, {}).get(horizon, {})
                ml_ret_pct = (ml.get("predicted_return", 0.0) or 0.0) * 100.0
                s_ml = _forecast_to_score(ml_ret_pct)

                # ── Forecast ensemble (Step 1) ─────────────────────────────
                forecast_score = (
                    self.ens_w_tfm * s_tfm
                    + self.ens_w_chr * s_chr
                    + self.ens_w_ml  * s_ml
                )

                # ── Other signals ──────────────────────────────────────────
                s_tech = tech_scores.get(ticker, 0.5)
                s_sent = _sentiment_to_score(sentiment.get(ticker, 0.0))
                s_fund = fundamentals.get(ticker, 0.5)
                s_mac  = macro.get(ticker, 0.5)

                # ── Composite (Step 2) ─────────────────────────────────────
                composite = (
                    self.w_forecast     * forecast_score
                    + self.w_technical  * s_tech
                    + self.w_sentiment  * s_sent
                    + self.w_fundamentals * s_fund
                    + self.w_macro      * s_mac
                )
                scores_per_horizon[horizon] = round(float(composite), 4)

            scores_per_horizon["overall"] = round(
                float(np.mean([scores_per_horizon[h] for h in ("short", "medium", "long")])),
                4,
            )
            raw_scores[ticker] = scores_per_horizon

        fused = self._normalise_per_horizon(raw_scores)
        logger.info("Signal fusion complete")
        return fused

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_per_horizon(
        raw: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Min-max normalise scores across tickers per horizon."""
        horizons = ("short", "medium", "long", "overall")
        normalised = {ticker: {} for ticker in raw}
        for h in horizons:
            h_scores = {t: raw[t].get(h, 0.5) for t in raw}
            normed   = normalize_scores(h_scores)
            for t, v in normed.items():
                normalised[t][h] = round(v, 4)
        return normalised

    def _validate_weights(self) -> None:
        ens_total = self.ens_w_tfm + self.ens_w_chr + self.ens_w_ml
        if abs(ens_total - 1.0) > 1e-3:
            logger.warning(
                "Ensemble sub-weights sum to %.3f — auto-normalising", ens_total
            )
            self.ens_w_tfm /= ens_total
            self.ens_w_chr /= ens_total
            self.ens_w_ml  /= ens_total

        sig_total = (
            self.w_forecast + self.w_technical + self.w_sentiment
            + self.w_fundamentals + self.w_macro
        )
        if abs(sig_total - 1.0) > 1e-3:
            logger.warning(
                "Signal weights sum to %.3f — auto-normalising", sig_total
            )
            self.w_forecast     /= sig_total
            self.w_technical    /= sig_total
            self.w_sentiment    /= sig_total
            self.w_fundamentals /= sig_total
            self.w_macro        /= sig_total
