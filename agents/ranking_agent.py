"""
Ranking Agent
──────────────
Ranks tickers by composite score and produces three strategy portfolios
plus a diversified top-10 final pick.

Portfolios
  growth    — highest alpha_score (momentum + confidence)
  value     — highest mispricing (model > sector + market)
  defensive — lowest drawdown_risk with positive expected return

Diversity constraints
  • max 1 ticker per sector (from universe_agent sector map)
  • max 1 ticker per cluster (from clustering_agent)
  Both constraints applied via greedy round-robin selection.

Final top-10
  Merge growth / value / defensive deduplicated; re-rank by alpha_score.

Pre-ranking filters (configurable under `filters:`)
  • volatility_21d > max_annual_volatility  → skip
  • |TimesFM_pct - Chronos_pct| / 100 > max_model_disagreement → skip
  • ML confidence < min_confidence → skip

Scoring formula (when risk_scores available)
  rank_score = alpha_score (from risk engine)
Fallback (no risk engine)
  rank_score = predicted_return × confidence  (ML)
  rank_score = fused_score − 0.5             (no ML)
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
        port                = config.get("portfolio", {})
        self.growth_n       = int(port.get("growth_n",    5))
        self.value_n        = int(port.get("value_n",     5))
        self.defensive_n    = int(port.get("defensive_n", 5))
        self.final_n        = int(port.get("final_n",     10))

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
        risk_scores:        Optional[Dict[str, dict]] = None,
        cluster_map:        Optional[Dict[str, int]]  = None,
        sector_map:         Optional[Dict[str, str]]  = None,
    ) -> dict:
        """
        Returns
        -------
        {
            "1_month":  [ticker, ...],
            "6_month":  [ticker, ...],
            "1_year":   [ticker, ...],
            "growth":   [ticker, ...],
            "value":    [ticker, ...],
            "defensive":[ticker, ...],
            "final_top_10_diversified": [ticker, ...],
            "details":  {ticker: {...}},
            "metrics":  {"sector_distribution": {...}},
        }
        """
        fundamentals   = fundamentals   or {}
        macro          = macro          or {}
        ml_predictions = ml_predictions or {}
        risk_scores    = risk_scores    or {}
        cluster_map    = cluster_map    or {}
        sector_map     = sector_map     or {}

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

        eligible = [t for t in fused_scores if t not in filtered_out]

        # ── Horizon-based rankings (legacy) ───────────────────────────────────
        top_short  = self._rank_horizon(fused_scores, "short",  ml_predictions, filtered_out)
        top_medium = self._rank_horizon(fused_scores, "medium", ml_predictions, filtered_out)
        top_long   = self._rank_horizon(fused_scores, "long",   ml_predictions, filtered_out)

        # ── Strategy portfolios ───────────────────────────────────────────────
        growth_port    = self._build_growth(eligible, risk_scores, cluster_map, sector_map)
        value_port     = self._build_value(eligible, risk_scores, cluster_map, sector_map)
        defensive_port = self._build_defensive(eligible, risk_scores, featured_data, cluster_map, sector_map)

        # ── Final diversified top-10 ──────────────────────────────────────────
        final_top = self._build_final(
            growth_port, value_port, defensive_port,
            risk_scores, cluster_map, sector_map,
        )

        # ── Details for all mentioned tickers ────────────────────────────────
        interesting = set(top_short + top_medium + top_long + growth_port + value_port + defensive_port + final_top)
        details = {}
        for ticker in interesting:
            details[ticker] = self._build_detail(
                ticker, fused_scores, timesfm_preds, chronos_preds,
                tech_scores, sentiment_scores, featured_data,
                fundamentals, macro, ml_predictions, risk_scores, sector_map,
            )

        # ── Sector distribution of final top ─────────────────────────────────
        sector_dist: Dict[str, int] = {}
        for t in final_top:
            s = sector_map.get(t, "Unknown")
            sector_dist[s] = sector_dist.get(s, 0) + 1

        output = {
            "1_month":  top_short,
            "6_month":  top_medium,
            "1_year":   top_long,
            "growth":   growth_port,
            "value":    value_port,
            "defensive": defensive_port,
            "final_top_10_diversified": final_top,
            "details":  details,
            "metrics":  {"sector_distribution": sector_dist},
        }
        logger.info("Rankings — 1mo: %s | 6mo: %s | 1yr: %s", top_short, top_medium, top_long)
        logger.info("Growth: %s", growth_port)
        logger.info("Value: %s", value_port)
        logger.info("Defensive: %s", defensive_port)
        logger.info("Final top 10: %s", final_top)
        return output

    # ── Filters ───────────────────────────────────────────────────────────────

    def _apply_filters(
        self,
        timesfm_preds:  Dict[str, Dict[str, dict]],
        chronos_preds:  Dict[str, dict],
        featured_data:  Dict,
        ml_predictions: Dict[str, Dict[str, dict]],
    ) -> Set[str]:
        skip: Set[str] = set()
        all_tickers = set(timesfm_preds) | set(chronos_preds) | set(featured_data)

        for ticker in all_tickers:
            df = featured_data.get(ticker)
            if df is not None and "volatility_21d" in df.columns:
                vol_series = df["volatility_21d"].dropna()
                if not vol_series.empty and float(vol_series.iloc[-1]) > self.max_vol:
                    skip.add(ticker)
                    continue

            for h in ("short", "medium", "long"):
                tfm_pct = timesfm_preds.get(ticker, {}).get(h, {}).get("pct_change") or 0.0
                chr_pct = chronos_preds.get(ticker, {}).get(h, {}).get("pct_change") or 0.0
                if abs(tfm_pct - chr_pct) / 100.0 > self.max_disagree:
                    skip.add(ticker)
                    break

            if ticker in skip:
                continue

            if ml_predictions and ticker in ml_predictions:
                confs = [
                    ml_predictions[ticker].get(h, {}).get("confidence", 0.5)
                    for h in ("short", "medium", "long")
                ]
                if max(confs) < self.min_confidence:
                    skip.add(ticker)

        return skip

    # ── Horizon ranking (legacy) ──────────────────────────────────────────────

    def _rank_horizon(
        self,
        fused_scores:   Dict[str, Dict[str, float]],
        horizon:        str,
        ml_predictions: Dict[str, Dict[str, dict]],
        filtered_out:   Set[str],
    ) -> List[str]:
        scored = []
        for ticker, scores in fused_scores.items():
            if ticker in filtered_out:
                continue
            if ml_predictions and ticker in ml_predictions:
                ml  = ml_predictions[ticker].get(horizon, {})
                pred_ret = ml.get("predicted_return", 0.0) or 0.0
                conf     = ml.get("confidence", 0.5) or 0.5
                rank_score = pred_ret * conf
            else:
                rank_score = (scores.get(horizon, 0.5) or 0.5) - 0.5
            scored.append((ticker, rank_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[: self.top_n]]

    # ── Strategy portfolio builders ───────────────────────────────────────────

    def _build_growth(
        self,
        eligible:    List[str],
        risk_scores: Dict[str, dict],
        cluster_map: Dict[str, int],
        sector_map:  Dict[str, str],
    ) -> List[str]:
        """Highest alpha_score (return * confidence / vol)."""
        def key(t):
            return risk_scores.get(t, {}).get("alpha_score", 0.0)
        ranked = sorted(eligible, key=key, reverse=True)
        return self._diverse_select(ranked, cluster_map, sector_map, self.growth_n)

    def _build_value(
        self,
        eligible:    List[str],
        risk_scores: Dict[str, dict],
        cluster_map: Dict[str, int],
        sector_map:  Dict[str, str],
    ) -> List[str]:
        """Highest mispricing (model predicts more than sector + market)."""
        def key(t):
            return risk_scores.get(t, {}).get("mispricing", 0.0)
        ranked = sorted(eligible, key=key, reverse=True)
        return self._diverse_select(ranked, cluster_map, sector_map, self.value_n)

    def _build_defensive(
        self,
        eligible:     List[str],
        risk_scores:  Dict[str, dict],
        featured_data: Dict,
        cluster_map:  Dict[str, int],
        sector_map:   Dict[str, str],
    ) -> List[str]:
        """Positive expected return, sorted by lowest drawdown risk."""
        candidates = [
            t for t in eligible
            if (risk_scores.get(t, {}).get("expected_return", 0.0) or 0.0) > 0
        ]
        if not candidates:
            candidates = eligible[:]  # relax if nothing passes
        def key(t):
            return risk_scores.get(t, {}).get("drawdown_risk", 1.0)
        ranked = sorted(candidates, key=key)  # ascending — less drawdown = better
        return self._diverse_select(ranked, cluster_map, sector_map, self.defensive_n)

    def _build_final(
        self,
        growth:      List[str],
        value:       List[str],
        defensive:   List[str],
        risk_scores: Dict[str, dict],
        cluster_map: Dict[str, int],
        sector_map:  Dict[str, str],
    ) -> List[str]:
        """
        Combine all three strategy lists, deduplicate, re-rank by alpha_score,
        apply diversity constraints, return top final_n.
        """
        combined = list(dict.fromkeys(growth + value + defensive))  # preserve order, dedup
        def key(t):
            return risk_scores.get(t, {}).get("alpha_score", 0.0)
        ranked = sorted(combined, key=key, reverse=True)
        return self._diverse_select(ranked, cluster_map, sector_map, self.final_n)

    # ── Diversity selection ───────────────────────────────────────────────────

    @staticmethod
    def _diverse_select(
        ranked:      List[str],
        cluster_map: Dict[str, int],
        sector_map:  Dict[str, str],
        n:           int,
    ) -> List[str]:
        """Greedy selection: skip a ticker if its cluster or sector is already represented."""
        selected:        List[str] = []
        used_clusters:   Set[int]  = set()
        used_sectors:    Set[str]  = set()

        for ticker in ranked:
            if len(selected) >= n:
                break
            cluster = cluster_map.get(ticker)
            sector  = sector_map.get(ticker, "Unknown")

            if cluster is not None and cluster in used_clusters:
                continue
            if sector != "Unknown" and sector in used_sectors:
                continue

            selected.append(ticker)
            if cluster is not None:
                used_clusters.add(cluster)
            used_sectors.add(sector)

        # Relax constraints if we couldn't fill the list
        if len(selected) < n:
            for ticker in ranked:
                if ticker not in selected:
                    selected.append(ticker)
                if len(selected) >= n:
                    break

        return selected[:n]

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
        risk_scores:      Dict[str, dict],
        sector_map:       Dict[str, str],
    ) -> dict:
        scores = fused_scores.get(ticker, {})
        tfm    = timesfm_preds.get(ticker, {})
        chr_   = chronos_preds.get(ticker, {})
        ml     = ml_predictions.get(ticker, {})
        risk   = risk_scores.get(ticker, {})

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
            "sector":             sector_map.get(ticker, "Unknown"),
            # Risk engine outputs
            "alpha_score":        risk.get("alpha_score",    0.0),
            "mispricing":         risk.get("mispricing",     0.0),
            "sharpe_proxy":       risk.get("sharpe_proxy",   0.0),
            "drawdown_risk":      risk.get("drawdown_risk",  0.0),
            "expected_return":    risk.get("expected_return", 0.0),
            "volatility":         risk.get("volatility",     0.0),
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

        confs = [ml.get(h, {}).get("confidence", 0.5) for h in ("short", "medium", "long")]
        detail["ml_confidence"] = round(float(np.mean(confs)), 4)

        df = featured_data.get(ticker)
        if df is not None and not df.empty:
            detail["latest_price"] = round(float(df["Close"].iloc[-1]), 2)
            detail["latest_date"]  = str(df.index[-1])[:10]

        return detail
