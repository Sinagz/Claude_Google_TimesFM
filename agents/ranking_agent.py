"""
Ranking Agent
──────────────
Sorts tickers by fused composite score and returns the top-N for each
forecast horizon (short / medium / long).

Also builds the enriched detail block for each top ticker that appears
in the final results.json output.
"""

from typing import Dict, List

from utils.helpers import setup_logger

logger = setup_logger("ranking_agent")


class RankingAgent:
    def __init__(self, config: dict):
        self.top_n = config["ranking"]["top_n"]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        fused_scores:       Dict[str, Dict[str, float]],
        timesfm_preds:      Dict[str, Dict[str, dict]],
        chronos_preds:      Dict[str, dict],
        tech_scores:        Dict[str, float],
        sentiment_scores:   Dict[str, float],
        featured_data:      Dict,
        fundamentals:       Dict[str, float] = None,
        macro:              Dict[str, float] = None,
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
        logger.info("Ranking %d tickers …", len(fused_scores))

        top_short  = self._rank(fused_scores, "short")
        top_medium = self._rank(fused_scores, "medium")
        top_long   = self._rank(fused_scores, "long")

        fundamentals = fundamentals or {}
        macro        = macro        or {}

        # Build detail records only for tickers appearing in any top list
        interesting = set(top_short + top_medium + top_long)
        details = {}
        for ticker in interesting:
            details[ticker] = self._build_detail(
                ticker,
                fused_scores,
                timesfm_preds,
                chronos_preds,
                tech_scores,
                sentiment_scores,
                featured_data,
                fundamentals,
                macro,
            )

        output = {
            "1_month": top_short,
            "6_month": top_medium,
            "1_year":  top_long,
            "details": details,
        }

        logger.info(
            "Rankings done — 1mo: %s | 6mo: %s | 1yr: %s",
            top_short, top_medium, top_long,
        )
        return output

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rank(self, fused: Dict[str, Dict[str, float]], horizon: str) -> List[str]:
        """Sort tickers by *horizon* score descending; return top N."""
        scored = [(t, fused[t].get(horizon, 0.0)) for t in fused]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[: self.top_n]]

    @staticmethod
    def _build_detail(
        ticker: str,
        fused_scores:       Dict[str, Dict[str, float]],
        timesfm_preds:      Dict[str, Dict[str, dict]],
        chronos_preds:      Dict[str, dict],
        tech_scores:        Dict[str, float],
        sentiment_scores:   Dict[str, float],
        featured_data:      Dict,
        fundamentals:       Dict[str, float] = None,
        macro:              Dict[str, float] = None,
    ) -> dict:
        scores = fused_scores.get(ticker, {})
        tfm    = timesfm_preds.get(ticker, {})
        chr_   = chronos_preds.get(ticker, {})

        detail: dict = {
            "score":              scores.get("overall", 0.0),
            "score_1month":       scores.get("short",   0.0),
            "score_6month":       scores.get("medium",  0.0),
            "score_1year":        scores.get("long",    0.0),
            "sentiment":          sentiment_scores.get(ticker, 0.0),
            "technical_score":    tech_scores.get(ticker, 0.5),
            "fundamentals_score": (fundamentals or {}).get(ticker, 0.5),
            "macro_score":        (macro or {}).get(ticker, 0.5),
            "agreement_score":    chr_.get("agreement_score", 0.5),
            "divergence_warning": chr_.get("divergence_warning", False),
        }

        # TimesFM predictions
        for h_name, h_key in [("short", "1month"), ("medium", "6month"), ("long", "1year")]:
            pred = tfm.get(h_name, {})
            detail[f"timesfm_{h_key}"] = {
                "point":      pred.get("point"),
                "low":        pred.get("low"),
                "high":       pred.get("high"),
                "pct_change": pred.get("pct_change"),
            }

        # Chronos predictions
        for h_name, h_key in [("short", "1month"), ("medium", "6month"), ("long", "1year")]:
            pred = chr_.get(h_name, {})
            detail[f"chronos_{h_key}"] = {
                "point":      pred.get("point"),
                "low":        pred.get("low"),
                "high":       pred.get("high"),
                "pct_change": pred.get("pct_change"),
            }

        # Latest price
        df = featured_data.get(ticker)
        if df is not None and not df.empty:
            detail["latest_price"] = round(float(df["Close"].iloc[-1]), 2)
            detail["latest_date"]  = str(df.index[-1])[:10]

        return detail
