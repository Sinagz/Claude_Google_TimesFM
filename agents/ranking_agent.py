"""
Ranking Agent
──────────────
Ranks tickers by composite score and returns the top-N for each horizon.

Scoring formula (when ML model is available):
  score = predicted_return × confidence

Pre-ranking filters (configurable in config.yaml under `filters:`):
  • Tickers with volatility_21d > max_annual_volatility are excluded.
  • Tickers where |TimesFM_pct_change - Chronos_pct_change| > max_model_disagreement
    (per horizon) are excluded.
  • Tickers where ML confidence < min_confidence are excluded.

Falls back to fused composite score when ML predictions are unavailable.
"""

import math
from typing import Dict, List, Optional, Set

import numpy as np

from utils.helpers import setup_logger

logger = setup_logger("ranking_agent")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RankingAgent:
    def __init__(self, config: dict):
        self.top_n          = config["ranking"]["top_n"]
        self.config         = config
        flt                 = config.get("filters", {})
        self.min_confidence = flt.get("min_confidence",           0.40)
        self.max_vol        = flt.get("max_annual_volatility",    0.80)
        self.max_disagree   = flt.get("max_model_disagreement",   0.20)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        fused_scores:       Dict[str, Dict[str, float]],
        timesfm_preds:      Dict[str, Dict[str, dict]],
        chronos_preds:      Dict[str, dict],
        tech_scores:        Dict[str, float],
        sentiment_scores:   Dict[str, float],
        featured_data:      Dict,
        fundamentals:       Optional[Dict[str, float]] = None,
        macro:              Optional[Dict[str, float]] = None,
        ml_predictions:     Optional[Dict[str, Dict[str, dict]]] = None,
    ) -> dict:
        """
        Returns
        -------
        {
            "1_month":  [ticker, ...],
            "6_month":  [ticker, ...],
            "1_year":   [ticker, ...],
            "details":  {ticker: {...}}
        }
        """
        fundamentals   = fundamentals   or {}
        macro          = macro          or {}
        ml_predictions = ml_predictions or {}

        logger.info("Ranking %d tickers …", len(fused_scores))

        # Pre-ranking filters
        filtered_out = self._apply_filters(
            timesfm_preds, chronos_preds, featured_data, ml_predictions
        )
        if filtered_out:
            logger.info(
                "Pre-rank filters removed %d ticker(s): %s",
                len(filtered_out), sorted(filtered_out),
            )

        top_short  = self._rank(fused_scores, "short",  ml_predictions, filtered_out)
        top_medium = self._rank(fused_scores, "medium", ml_predictions, filtered_out)
        top_long   = self._rank(fused_scores, "long",   ml_predictions, filtered_out)

        interesting = set(top_short + top_medium + top_long)
        details = {}
        for ticker in interesting:
            details[ticker] = self._build_detail(
                ticker, fused_scores, timesfm_preds, chronos_preds,
                tech_scores, sentiment_scores, featured_data,
                fundamentals, macro, ml_predictions,
            )

        output = {
            "1_month": top_short,
            "6_month": top_medium,
            "1_year":  top_long,
            "details": details,
        }
        logger.info(
            "Rankings — 1mo: %s | 6mo: %s | 1yr: %s",
            top_short, top_medium, top_long,
        )
        return output

    # ── Filters ───────────────────────────────────────────────────────────────

    def _apply_filters(
        self,
        timesfm_preds:  Dict[str, Dict[str, dict]],
        chronos_preds:  Dict[str, dict],
        featured_data:  Dict,
        ml_predictions: Dict[str, Dict[str, dict]],
    ) -> Set[str]:
        """Return set of tickers that fail at least one filter."""
        skip: Set[str] = set()
        all_tickers = (
            set(timesfm_preds) | set(chronos_preds) | set(featured_data)
        )

        for ticker in all_tickers:
            # ── Volatility filter ──────────────────────────────────────────
            df = featured_data.get(ticker)
            if df is not None and "volatility_21d" in df.columns:
                vol_series = df["volatility_21d"].dropna()
                if not vol_series.empty:
                    if float(vol_series.iloc[-1]) > self.max_vol:
                        skip.add(ticker)
                        continue

            # ── Model disagreement filter (any horizon) ────────────────────
            for h in ("short", "medium", "long"):
                tfm_pct = timesfm_preds.get(ticker, {}).get(h, {}).get("pct_change") or 0.0
                chr_pct = chronos_preds.get(ticker, {}).get(h, {}).get("pct_change") or 0.0
                # pct_change values are in % (e.g. 5.0 means +5%)
                if abs(tfm_pct - chr_pct) / 100.0 > self.max_disagree:
                    skip.add(ticker)
                    break

            if ticker in skip:
                continue

            # ── ML confidence filter ───────────────────────────────────────
            if ml_predictions and ticker in ml_predictions:
                confidences = [
                    ml_predictions[ticker].get(h, {}).get("confidence", 0.5)
                    for h in ("short", "medium", "long")
                ]
                if max(confidences) < self.min_confidence:
                    skip.add(ticker)

        return skip

    # ── Ranking ───────────────────────────────────────────────────────────────

    def _rank(
        self,
        fused_scores:  Dict[str, Dict[str, float]],
        horizon:       str,
        ml_predictions: Dict[str, Dict[str, dict]],
        filtered_out:  Set[str],
    ) -> List[str]:
        """Sort eligible tickers by score descending; return top N."""
        scored = []

        for ticker, scores in fused_scores.items():
            if ticker in filtered_out:
                continue

            if ml_predictions and ticker in ml_predictions:
                ml = ml_predictions[ticker].get(horizon, {})
                pred_ret = ml.get("predicted_return", 0.0) or 0.0
                conf     = ml.get("confidence", 0.5) or 0.5
                # Primary rank key: expected return weighted by confidence
                rank_score = pred_ret * conf
            else:
                # Fallback: fused score shifted to be centred around 0
                rank_score = (scores.get(horizon, 0.5) or 0.5) - 0.5

            scored.append((ticker, rank_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[: self.top_n]]

    # ── Detail builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_detail(
        ticker:           str,
        fused_scores:     Dict[str, Dict[str, float]],
        timesfm_preds:    Dict[str, Dict[str, dict]],
        chronos_preds:    Dict[str, dict],
        tech_scores:      Dict[str, float],
        sentiment_scores: Dict[str, float],
        featured_data:    Dict,
        fundamentals:     Dict[str, float],
        macro:            Dict[str, float],
        ml_predictions:   Dict[str, Dict[str, dict]],
    ) -> dict:
        scores = fused_scores.get(ticker, {})
        tfm    = timesfm_preds.get(ticker, {})
        chr_   = chronos_preds.get(ticker, {})
        ml     = ml_predictions.get(ticker, {})

        detail: dict = {
            "score":              scores.get("overall", 0.0),
            "score_1month":       scores.get("short",   0.0),
            "score_6month":       scores.get("medium",  0.0),
            "score_1year":        scores.get("long",    0.0),
            "sentiment":          sentiment_scores.get(ticker, 0.0),
            "technical_score":    tech_scores.get(ticker, 0.5),
            "fundamentals_score": fundamentals.get(ticker, 0.5),
            "macro_score":        macro.get(ticker, 0.5),
            "agreement_score":    chr_.get("agreement_score", 0.5),
            "divergence_warning": chr_.get("divergence_warning", False),
        }

        # TimesFM predictions
        for h_name, h_key in (("short", "1month"), ("medium", "6month"), ("long", "1year")):
            pred = tfm.get(h_name, {})
            detail[f"timesfm_{h_key}"] = {
                "point":      pred.get("point"),
                "low":        pred.get("low"),
                "high":       pred.get("high"),
                "pct_change": pred.get("pct_change"),
            }

        # Chronos predictions
        for h_name, h_key in (("short", "1month"), ("medium", "6month"), ("long", "1year")):
            pred = chr_.get(h_name, {})
            detail[f"chronos_{h_key}"] = {
                "point":      pred.get("point"),
                "low":        pred.get("low"),
                "high":       pred.get("high"),
                "pct_change": pred.get("pct_change"),
            }

        # ML meta-model predictions
        for h_name, h_key in (("short", "1month"), ("medium", "6month"), ("long", "1year")):
            pred = ml.get(h_name, {})
            detail[f"ml_{h_key}"] = {
                "predicted_return": pred.get("predicted_return", 0.0),
                "confidence":       pred.get("confidence", 0.5),
            }

        # Overall confidence = mean of ML confidences
        confs = [ml.get(h, {}).get("confidence", 0.5) for h in ("short", "medium", "long")]
        detail["ml_confidence"] = round(float(np.mean(confs)), 4)

        # Latest price
        df = featured_data.get(ticker)
        if df is not None and not df.empty:
            detail["latest_price"] = round(float(df["Close"].iloc[-1]), 2)
            detail["latest_date"]  = str(df.index[-1])[:10]

        return detail
