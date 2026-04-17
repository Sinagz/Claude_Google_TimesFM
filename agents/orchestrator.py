"""
Orchestrator Agent — Production Quant Trading Platform
────────────────────────────────────────────────────────
Full 14-stage pipeline:

  0  UniverseAgent          → pre-screen 700+ tickers → ~150 candidates
  1  DataAgent              → raw OHLCV per ticker
  2  FeatureAgent           → enriched DataFrames, tech scores, price arrays
  3  TimesFMAgent       *   → deep-learning price forecasts (3 horizons)
  4  ChronosAgent       *   → probabilistic forecasts + agreement
  5  SentimentAgent         → news sentiment (time-decay weighted)
  5b FundamentalsAgent      → P/E, EPS growth, fundamental scores
  5c MacroAgent             → VIX, yield curve, macro regime score
  6  CrashDetectionAgent    → crash_probability, market regime
  7  MLModel (train+predict)→ XGBoost meta-model on technical features
  8  FusionAgent            → 3-model ensemble + signal fusion
  8b ClusteringAgent        → KMeans for diversification constraints
  8c RiskEngine             → alpha_score, mispricing, regime-adjusted risk
  9  RLTradingAgent         → PPO policy: portfolio weights
  10 RankingAgent           → growth / value / defensive / final-top-10
  11 IBKRExecutionAgent     → paper/live order execution (optional)
  12 BacktestAgent          → walk-forward with costs/slippage/RL sizing

Stages marked * are CRITICAL — exceptions propagate and abort the run.
All other stages fail gracefully (logged, pipeline continues).
"""

import time
from typing import Dict, List, Optional

from agents.backtest_agent       import BacktestAgent
from agents.chronos_agent        import ChronosAgent
from agents.clustering_agent     import ClusteringAgent
from agents.crash_detection_agent import CrashDetectionAgent
from agents.data_agent           import DataAgent
from agents.feature_agent        import FeatureAgent
from agents.fundamentals_agent   import FundamentalsAgent
from agents.fusion_agent         import FusionAgent
from agents.ibkr_execution_agent import IBKRExecutionAgent
from agents.macro_agent          import MacroAgent
from agents.ranking_agent        import RankingAgent
from agents.risk_engine          import RiskEngine
from agents.rl_trading_agent     import RLTradingAgent
from agents.sentiment_agent      import SentimentAgent
from agents.timesfm_agent        import TimesFMAgent
from agents.universe_agent       import UniverseAgent, get_ticker_sector
from models.ml_model             import MLModel
from utils.helpers               import ensure_dirs, get_all_tickers, save_json, setup_logger

logger = setup_logger("orchestrator")


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        ensure_dirs(config)

        self.universe_agent       = UniverseAgent(config)
        self.data_agent           = DataAgent(config)
        self.feature_agent        = FeatureAgent(config)
        self.timesfm_agent        = TimesFMAgent(config)
        self.chronos_agent        = ChronosAgent(config)
        self.sentiment_agent      = SentimentAgent(config)
        self.fundamentals_agent   = FundamentalsAgent(config)
        self.macro_agent          = MacroAgent(config)
        self.crash_agent          = CrashDetectionAgent(config)
        self.ml_model             = MLModel(config)
        self.fusion_agent         = FusionAgent(config)
        self.clustering_agent     = ClusteringAgent(config)
        self.risk_engine          = RiskEngine(config)
        self.rl_agent             = RLTradingAgent(config)
        self.ranking_agent        = RankingAgent(config)
        self.ibkr_agent           = IBKRExecutionAgent(config)
        self.backtest_agent       = BacktestAgent(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        t_start = time.time()
        logger.info("=" * 70)
        logger.info("  Production Quant Trading Platform — Pipeline Started")
        logger.info("=" * 70)

        # ── Stage 0: Universe Pre-Screening ──────────────────────────────
        universe_result = self._run_stage("universe_prescreening", self.universe_agent.run)
        if universe_result:
            tickers_to_fetch: Optional[List[str]] = list(universe_result.keys())
            sector_map: Dict[str, str] = {
                t: d.get("sector", get_ticker_sector(t))
                for t, d in universe_result.items()
            }
            logger.info("Universe yielded %d candidates", len(tickers_to_fetch))
        else:
            tickers_to_fetch = None
            sector_map       = {t: get_ticker_sector(t) for t in get_all_tickers(self.config)}
            logger.info("Universe agent skipped — using config tickers")

        # ── Stage 1: Data Ingestion ───────────────────────────────────────
        raw_data = self._run_stage("data_ingestion", self.data_agent.run, tickers_to_fetch)
        if not raw_data:
            logger.error("Data ingestion returned nothing — aborting")
            return {}
        self.raw_data = raw_data
        for t in raw_data:
            if t not in sector_map:
                sector_map[t] = get_ticker_sector(t)

        # ── Stage 2: Feature Engineering ─────────────────────────────────
        feat_result = self._run_stage("feature_engineering", self.feature_agent.run, raw_data)
        featured_data, tech_scores, price_arrays = feat_result if feat_result else ({}, {}, {})
        tickers = list(price_arrays.keys()) or get_all_tickers(self.config)

        # ── Stage 3: TimesFM Forecasting  [CRITICAL] ─────────────────────
        timesfm_preds = self._run_critical_stage(
            "timesfm_forecast", self.timesfm_agent.run, price_arrays
        )

        # ── Stage 4: Chronos Verification  [CRITICAL] ────────────────────
        chronos_preds = self._run_critical_stage(
            "chronos_verification", self.chronos_agent.run, price_arrays, timesfm_preds
        )

        # ── Stage 5: Sentiment ────────────────────────────────────────────
        sentiment = self._run_stage(
            "sentiment_analysis", self.sentiment_agent.run, tickers
        ) or {t: 0.0 for t in tickers}

        # ── Stage 5b: Fundamentals ────────────────────────────────────────
        fundamentals = self._run_stage(
            "fundamentals", self.fundamentals_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── Stage 5c: Macro ───────────────────────────────────────────────
        macro = self._run_stage(
            "macro_context", self.macro_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── Stage 6: Crash Detection ──────────────────────────────────────
        crash_result = self._run_stage(
            "crash_detection", self.crash_agent.run, featured_data, raw_data
        ) or {"crash_probability": 0.0, "regime": "neutral",
              "signals": {}, "recommendation": "HOLD"}
        crash_prob = float(crash_result.get("crash_probability", 0.0))
        regime     = crash_result.get("regime", "neutral")
        recommendation = crash_result.get("recommendation", "HOLD")

        # ── Stage 7: ML Meta-Model ────────────────────────────────────────
        ml_predictions: Dict = {}
        if featured_data:
            trained = self._run_stage("ml_training", self.ml_model.train, featured_data)
            if trained:
                ml_predictions = self._run_stage(
                    "ml_inference", self.ml_model.predict, featured_data
                ) or {}

        # ── Stage 8: Signal Fusion ────────────────────────────────────────
        fused_scores = self._run_stage(
            "signal_fusion",
            self.fusion_agent.run,
            timesfm_preds, chronos_preds, tech_scores,
            sentiment, fundamentals, macro, ml_predictions,
        ) or {}

        # ── Stage 8b: Clustering ──────────────────────────────────────────
        cluster_result = self._run_stage(
            "clustering", self.clustering_agent.run, featured_data
        ) or {"cluster_map": {}, "cluster_members": {}, "n_clusters": 0}
        cluster_map = cluster_result.get("cluster_map", {})

        # ── Stage 8c: Risk Engine ─────────────────────────────────────────
        risk_result = self._run_stage(
            "risk_engine", self.risk_engine.run,
            featured_data, ml_predictions, timesfm_preds, chronos_preds,
        ) or {"scores": {}, "regime": regime}
        risk_scores = risk_result.get("scores", {})

        # ── Stage 9: RL Trading Agent ─────────────────────────────────────
        rl_trained = self._run_stage(
            "rl_training", self.rl_agent.train,
            raw_data, featured_data, ml_predictions, fused_scores, crash_prob,
        )
        rl_portfolio = self._run_stage(
            "rl_inference", self.rl_agent.predict,
            featured_data, ml_predictions, fused_scores, crash_prob,
        ) or {"weights": {}, "cash_weight": 1.0,
               "recommended_action": recommendation, "confidence": 0.5}

        rl_weights = rl_portfolio.get("weights", {})

        # ── Stage 10: Ranking ─────────────────────────────────────────────
        ranking_output = self._run_stage(
            "ranking", self.ranking_agent.run,
            fused_scores, timesfm_preds, chronos_preds,
            tech_scores, sentiment, featured_data,
            fundamentals, macro, ml_predictions,
            risk_scores, cluster_map, sector_map,
        ) or {
            "1_month": [], "6_month": [], "1_year": [],
            "growth": [], "value": [], "defensive": [],
            "final_top_10_diversified": [],
            "details": {}, "metrics": {"sector_distribution": {}},
        }

        # ── Stage 11: IBKR Execution (optional) ──────────────────────────
        execution_result = self._run_stage(
            "ibkr_execution", self.ibkr_agent.run,
            rl_portfolio, ranking_output, crash_result, featured_data,
        ) or {"executed_trades": [], "status": "disabled", "message": ""}

        # ── Stage 12: Backtest ────────────────────────────────────────────
        backtest_result = self._run_stage(
            "backtest", self.backtest_agent.run,
            raw_data, ml_predictions, regime, rl_weights,
        ) or {}

        # ── Compute expected return from risk scores ───────────────────────
        top_10 = ranking_output.get("final_top_10_diversified", [])
        exp_ret = 0.0
        if top_10 and risk_scores:
            vals = [risk_scores.get(t, {}).get("expected_return", 0.0) for t in top_10]
            vals = [v for v in vals if v is not None]
            exp_ret = float(np.mean(vals)) if vals else 0.0

        # ── Assemble output ───────────────────────────────────────────────
        elapsed = round(time.time() - t_start, 1)
        output = {
            # Rankings
            "1_month":   ranking_output["1_month"],
            "6_month":   ranking_output["6_month"],
            "1_year":    ranking_output["1_year"],
            "growth":    ranking_output.get("growth",    []),
            "value":     ranking_output.get("value",     []),
            "defensive": ranking_output.get("defensive", []),
            # Hedge fund outputs
            "top_10_stocks":              top_10,
            "final_top_10_diversified":   top_10,
            "rl_portfolio":               rl_portfolio,
            "crash_risk":                 round(crash_prob, 4),
            "market_regime":              regime,
            "expected_return":            f"{exp_ret:+.2%}",
            "recommendation":             rl_portfolio.get("recommended_action", recommendation),
            # Risk metrics
            "risk_metrics": {
                "sharpe":        backtest_result.get("sharpe_ratio",  0.0),
                "sortino":       backtest_result.get("sortino_ratio", 0.0),
                "drawdown":      backtest_result.get("max_drawdown",  0.0),
                "cagr":          backtest_result.get("cagr",          0.0),
                "win_rate":      backtest_result.get("win_rate",      0.0),
            },
            # Execution
            "executed_trades": execution_result.get("executed_trades", []),
            # Details + backtest
            "details":   ranking_output["details"],
            "backtest":  backtest_result,
            "metrics": {
                **ranking_output.get("metrics", {}),
                "regime":        regime,
                "crash_signals": crash_result.get("signals", {}),
                "n_clusters":    cluster_result.get("n_clusters", 0),
            },
            "run_metadata": {
                "elapsed_seconds":      elapsed,
                "n_tickers_analysed":   len(price_arrays),
                "n_universe_screened":  len(tickers_to_fetch) if tickers_to_fetch else len(raw_data),
                "models_used":          ["TimesFM-2.5", "Chronos-Bolt-Small", "XGBoost-ML", "PPO-RL"],
                "ml_trained":           bool(ml_predictions),
                "rl_trained":           bool(rl_trained),
                "regime":               regime,
                "crash_probability":    round(crash_prob, 4),
            },
        }

        save_json(output, self.config["output_path"])
        logger.info("=" * 70)
        logger.info("Pipeline complete in %.1f s  →  %s", elapsed, self.config["output_path"])
        logger.info("Regime: %-18s  Crash prob: %.3f", regime, crash_prob)
        logger.info("Recommendation: %s", output["recommendation"])
        logger.info("Top-10 (diversified): %s", top_10)
        logger.info("RL allocation (top-5): %s",
                    sorted(rl_weights.items(), key=lambda x: -x[1])[:5])
        logger.info("=" * 70)
        return output

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_critical_stage(self, name: str, fn, *args, **kwargs):
        logger.info("▶  [CRITICAL] %s", name)
        t0 = time.time()
        result = fn(*args, **kwargs)
        logger.info("✓  %s  (%.1f s)", name, time.time() - t0)
        return result

    def _run_stage(self, name: str, fn, *args, **kwargs):
        logger.info("▶  %s", name)
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            logger.info("✓  %s  (%.1f s)", name, time.time() - t0)
            return result
        except Exception as exc:
            logger.error("✗  %s FAILED: %s", name, exc, exc_info=True)
            return None


# ── Convenience import for numpy in output assembly ───────────────────────────
import numpy as np  # noqa: E402
