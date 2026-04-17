"""
Orchestrator Agent — Main Brain
─────────────────────────────────
Controls the full pipeline in dependency order.

Pipeline
  1  DataAgent           → raw OHLCV per ticker
  2  FeatureAgent        → enriched DataFrames, tech scores, price arrays,
                           market features (SPY returns, VIX)
  2b MLModel         *   → train on historical features; predict latest row
  3  TimesFMAgent    *   → price forecasts (3 horizons)          CRITICAL
  4  ChronosAgent    *   → independent forecasts + agreement     CRITICAL
  5  SentimentAgent      → news sentiment per ticker (decay-weighted)
  5b FundamentalsAgent   → fundamental scores per ticker
  5c MacroAgent          → macro regime score
  6  FusionAgent         → 3-model ensemble + signal fusion
  7  RankingAgent        → top-5 with filters + return×confidence scoring
  8  BacktestAgent       → walk-forward simulation, confidence-sized positions

Stages marked * are CRITICAL for the forecast pipeline. Non-critical stages
may fail gracefully without halting the run.
"""

import time
from typing import Dict

from agents.backtest_agent      import BacktestAgent
from agents.chronos_agent       import ChronosAgent
from agents.data_agent          import DataAgent
from agents.feature_agent       import FeatureAgent
from agents.fundamentals_agent  import FundamentalsAgent
from agents.fusion_agent        import FusionAgent
from agents.macro_agent         import MacroAgent
from agents.ranking_agent       import RankingAgent
from agents.sentiment_agent     import SentimentAgent
from agents.timesfm_agent       import TimesFMAgent
from models.ml_model             import MLModel
from utils.helpers               import ensure_dirs, get_all_tickers, save_json, setup_logger

logger = setup_logger("orchestrator")


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        ensure_dirs(config)

        self.data_agent         = DataAgent(config)
        self.feature_agent      = FeatureAgent(config)
        self.ml_model           = MLModel(config)
        self.timesfm_agent      = TimesFMAgent(config)
        self.chronos_agent      = ChronosAgent(config)
        self.sentiment_agent    = SentimentAgent(config)
        self.fundamentals_agent = FundamentalsAgent(config)
        self.macro_agent        = MacroAgent(config)
        self.fusion_agent       = FusionAgent(config)
        self.ranking_agent      = RankingAgent(config)
        self.backtest_agent     = BacktestAgent(config)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full pipeline and return the assembled results dict."""
        t_start = time.time()
        logger.info("=" * 60)
        logger.info("Pipeline started")
        logger.info("=" * 60)

        # ── 1. Data Ingestion ─────────────────────────────────────────────
        raw_data = self._run_stage("data_ingestion", self.data_agent.run)
        if not raw_data:
            logger.error("Data ingestion returned nothing — aborting")
            return {}
        self.raw_data = raw_data

        # ── 2. Feature Engineering ────────────────────────────────────────
        feat_result = self._run_stage(
            "feature_engineering", self.feature_agent.run, raw_data
        )
        if feat_result:
            featured_data, tech_scores, price_arrays = feat_result
        else:
            featured_data, tech_scores, price_arrays = {}, {}, {}

        tickers = list(price_arrays.keys()) or get_all_tickers(self.config)

        # ── 2b. ML Meta-Model (train + predict) ───────────────────────────
        ml_predictions: Dict = {}
        if featured_data:
            trained = self._run_stage(
                "ml_training", self.ml_model.train, featured_data
            )
            if trained:
                ml_predictions = self._run_stage(
                    "ml_inference", self.ml_model.predict, featured_data
                ) or {}

        # ── 3. TimesFM Forecasting  [CRITICAL] ───────────────────────────
        timesfm_preds = self._run_critical_stage(
            "timesfm_forecast", self.timesfm_agent.run, price_arrays
        )

        # ── 4. Chronos Verification  [CRITICAL] ──────────────────────────
        chronos_preds = self._run_critical_stage(
            "chronos_verification",
            self.chronos_agent.run,
            price_arrays,
            timesfm_preds,
        )

        # ── 5. Sentiment Analysis ─────────────────────────────────────────
        sentiment = self._run_stage(
            "sentiment_analysis", self.sentiment_agent.run, tickers
        ) or {t: 0.0 for t in tickers}

        # ── 5b. Fundamentals ──────────────────────────────────────────────
        fundamentals = self._run_stage(
            "fundamentals", self.fundamentals_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── 5c. Macro Context ─────────────────────────────────────────────
        macro = self._run_stage(
            "macro_context", self.macro_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── 6. Signal Fusion ──────────────────────────────────────────────
        fused_scores = self._run_stage(
            "signal_fusion",
            self.fusion_agent.run,
            timesfm_preds,
            chronos_preds,
            tech_scores,
            sentiment,
            fundamentals,
            macro,
            ml_predictions,
        ) or {}

        # ── 7. Ranking ────────────────────────────────────────────────────
        ranking_output = self._run_stage(
            "ranking",
            self.ranking_agent.run,
            fused_scores,
            timesfm_preds,
            chronos_preds,
            tech_scores,
            sentiment,
            featured_data,
            fundamentals,
            macro,
            ml_predictions,
        ) or {"1_month": [], "6_month": [], "1_year": [], "details": {}}

        # ── 8. Backtesting ────────────────────────────────────────────────
        backtest_result = self._run_stage(
            "backtest", self.backtest_agent.run, raw_data, ml_predictions
        ) or {}

        # ── Assemble output ───────────────────────────────────────────────
        elapsed = round(time.time() - t_start, 1)
        output = {
            "1_month":  ranking_output["1_month"],
            "6_month":  ranking_output["6_month"],
            "1_year":   ranking_output["1_year"],
            "details":  ranking_output["details"],
            "backtest": backtest_result,
            "run_metadata": {
                "elapsed_seconds":     elapsed,
                "n_tickers_analysed":  len(price_arrays),
                "data_sources":        ["yfinance", "alpha_vantage"],
                "models_used":         ["TimesFM-2.5", "Chronos-Bolt-Small", "XGBoost-ML"],
                "ml_trained":          bool(ml_predictions),
            },
        }

        save_json(output, self.config["output_path"])
        logger.info("=" * 60)
        logger.info(
            "Pipeline complete in %.1f s — results saved to %s",
            elapsed, self.config["output_path"],
        )
        logger.info("1-month  top 5: %s", output["1_month"])
        logger.info("6-month  top 5: %s", output["6_month"])
        logger.info("1-year   top 5: %s", output["1_year"])
        logger.info("=" * 60)
        return output

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_critical_stage(self, name: str, fn, *args, **kwargs):
        """Run a MUST-succeed stage. Any exception propagates and aborts pipeline."""
        logger.info("▶  Stage [CRITICAL]: %s", name)
        t0 = time.time()
        result = fn(*args, **kwargs)
        logger.info("✓  Stage %s done in %.1f s", name, time.time() - t0)
        return result

    def _run_stage(self, name: str, fn, *args, **kwargs):
        """Run a non-critical stage. Logs and swallows exceptions."""
        logger.info("▶  Stage: %s", name)
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            logger.info("✓  Stage %s done in %.1f s", name, time.time() - t0)
            return result
        except Exception as exc:
            logger.error("✗  Stage %s FAILED: %s", name, exc, exc_info=True)
            return None
