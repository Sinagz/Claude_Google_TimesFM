"""
Orchestrator Agent — Main Brain
─────────────────────────────────
Controls the full pipeline, runs agents in dependency order, and assembles
the final output dict.

Pipeline order
  1. DataAgent           → raw OHLCV per ticker
  2. FeatureAgent        → enriched DataFrames + tech scores + price arrays
  3. TimesFMAgent    *   → price forecasts (3 horizons)          CRITICAL
  4. ChronosAgent    *   → independent forecasts + agreement     CRITICAL
  5. SentimentAgent      → news sentiment per ticker
  5b. FundamentalsAgent  → fundamental scores per ticker
  5c. MacroAgent         → macro regime score (uniform)
  6. FusionAgent         → composite score per ticker × horizon
  7. RankingAgent        → top-5 lists + detail records
  8. BacktestAgent       → historical strategy simulation

Stages marked * are CRITICAL.  If TimesFM or Chronos cannot be loaded the
pipeline raises immediately and exits — these models are the core of the
system and there is no acceptable degraded mode.

Non-critical stages (sentiment, fundamentals, macro, backtest) may fail
without halting the run; they degrade to neutral defaults.
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
from utils.helpers               import ensure_dirs, get_all_tickers, save_json, setup_logger

logger = setup_logger("orchestrator")


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        ensure_dirs(config)

        # Instantiate all agents once; they hold any loaded models
        self.data_agent         = DataAgent(config)
        self.feature_agent      = FeatureAgent(config)
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
        """Execute the full pipeline and return the results dict."""
        t_start = time.time()
        logger.info("=" * 60)
        logger.info("Pipeline started")
        logger.info("=" * 60)

        # ── Stage 1: Data ─────────────────────────────────────────────────
        raw_data = self._run_stage("data_ingestion", self.data_agent.run)
        if not raw_data:
            logger.error("Data ingestion returned nothing — aborting")
            return {}
        self.raw_data = raw_data  # stored for dashboard access

        # ── Stage 2: Feature Engineering ────────────────────────────────
        feat_result = self._run_stage(
            "feature_engineering", self.feature_agent.run, raw_data
        )
        if feat_result:
            featured_data, tech_scores, price_arrays = feat_result
        else:
            featured_data, tech_scores, price_arrays = {}, {}, {}

        tickers = list(price_arrays.keys()) or get_all_tickers(self.config)

        # ── Stage 3: TimesFM Forecasting  [CRITICAL] ─────────────────────
        # Raises ModelLoadError → pipeline aborts if model unavailable.
        timesfm_preds = self._run_critical_stage(
            "timesfm_forecast", self.timesfm_agent.run, price_arrays
        )

        # ── Stage 4: Chronos Verification  [CRITICAL] ────────────────────
        # Raises ModelLoadError → pipeline aborts if model unavailable.
        chronos_preds = self._run_critical_stage(
            "chronos_verification",
            self.chronos_agent.run,
            price_arrays,
            timesfm_preds,
        )

        # ── Stage 5: Sentiment Analysis ───────────────────────────────────
        sentiment = self._run_stage(
            "sentiment_analysis", self.sentiment_agent.run, tickers
        ) or {t: 0.0 for t in tickers}

        # ── Stage 5b: Fundamentals ────────────────────────────────────────
        fundamentals = self._run_stage(
            "fundamentals", self.fundamentals_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── Stage 5c: Macro Context ───────────────────────────────────────
        macro = self._run_stage(
            "macro_context", self.macro_agent.run, tickers
        ) or {t: 0.5 for t in tickers}

        # ── Stage 6: Signal Fusion ────────────────────────────────────────
        fused_scores = self._run_stage(
            "signal_fusion",
            self.fusion_agent.run,
            timesfm_preds,
            chronos_preds,
            tech_scores,
            sentiment,
            fundamentals,
            macro,
        ) or {}

        # ── Stage 7: Ranking ──────────────────────────────────────────────
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
        ) or {"1_month": [], "6_month": [], "1_year": [], "details": {}}

        # ── Stage 8: Backtesting ──────────────────────────────────────────
        backtest_result = self._run_stage(
            "backtest", self.backtest_agent.run, raw_data
        ) or {}

        # ── Assemble final output ─────────────────────────────────────────
        elapsed = round(time.time() - t_start, 1)
        output = {
            "1_month":         ranking_output["1_month"],
            "6_month":         ranking_output["6_month"],
            "1_year":          ranking_output["1_year"],
            "details":         ranking_output["details"],
            "backtest":        backtest_result,
            "run_metadata": {
                "elapsed_seconds": elapsed,
                "n_tickers_analysed": len(price_arrays),
                "data_sources": ["yfinance", "alpha_vantage"],
                "models_used": ["TimesFM-2.5", "Chronos-Bolt-Small"],
            },
        }

        # Save to disk
        out_path = self.config["output_path"]
        save_json(output, out_path)
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f s — results saved to %s", elapsed, out_path)
        logger.info("1-month  top 5: %s", output["1_month"])
        logger.info("6-month  top 5: %s", output["6_month"])
        logger.info("1-year   top 5: %s", output["1_year"])
        logger.info("=" * 60)

        return output

    # ── Stage runners ─────────────────────────────────────────────────────────

    def _run_critical_stage(self, name: str, fn, *args, **kwargs):
        """
        Run a stage that MUST succeed.  Any exception propagates immediately,
        aborting the entire pipeline.  Used for TimesFM and Chronos.
        """
        logger.info("▶  Stage [CRITICAL]: %s", name)
        t0 = time.time()
        result = fn(*args, **kwargs)   # let exceptions propagate
        logger.info("✓  Stage %s done in %.1f s", name, time.time() - t0)
        return result

    def _run_stage(self, name: str, fn, *args, **kwargs):
        """
        Run a non-critical stage.  Logs and swallows exceptions so the
        pipeline continues with a None/empty result for this stage.
        """
        logger.info("▶  Stage: %s", name)
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            logger.info("✓  Stage %s done in %.1f s", name, time.time() - t0)
            return result
        except Exception as exc:
            logger.error("✗  Stage %s FAILED: %s", name, exc, exc_info=True)
            return None
