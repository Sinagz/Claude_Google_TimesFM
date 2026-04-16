"""
Chronos Verification Agent
───────────────────────────
Uses Amazon's Chronos Bolt (T5-based) probabilistic forecasting model to
independently forecast the same price series as TimesFM, then computes:

  • per-ticker agreement score  (1 = identical direction, 0 = opposite)
  • divergence warning flag     (True if predictions differ by > threshold)

Model source: https://huggingface.co/amazon/chronos-bolt-small
Package:      pip install chronos-forecasting

Bolt vs original Chronos: Bolt uses ChronosBoltPipeline (not ChronosPipeline).
The two classes share the same .predict() API but Bolt is ~50x faster.

HARD REQUIREMENT — no fallback
  Chronos is a core component of this system.  If the model cannot be
  loaded for any reason the agent raises a ModelLoadError and the entire
  pipeline is aborted.  There is intentionally no silent degradation path.
"""

from typing import Dict

import numpy as np

from utils.helpers import setup_logger

logger = setup_logger("chronos_agent")

_MIN_CONTEXT = 32
_DIVERGENCE_THRESHOLD = 0.10   # 10 % relative difference triggers warning


class ModelLoadError(RuntimeError):
    """Raised when a required forecasting model fails to load."""


class ChronosAgent:
    def __init__(self, config: dict):
        self.config = config
        self.model_id = config["model"]["chronos_checkpoint"]
        self.horizons = config["forecast"]["horizons"]
        self.device = self._resolve_device(config["model"]["device"])
        self._pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        price_arrays:       Dict[str, np.ndarray],
        timesfm_predictions: Dict[str, Dict[str, dict]],
    ) -> Dict[str, dict]:
        """
        Parameters
        ----------
        price_arrays        : {ticker -> Close-price array}
        timesfm_predictions : output from TimesFMAgent.run()

        Returns
        -------
        {
            ticker: {
                "short":  {"point": float, "pct_change": float},
                "medium": {...},
                "long":   {...},
                "agreement_score":   float,   # 0-1
                "divergence_warning": bool,
            }
        }

        Raises
        ------
        ModelLoadError  if Chronos cannot be loaded (aborts the pipeline).
        """
        self._ensure_pipeline_loaded()   # raises ModelLoadError on failure

        results: Dict[str, dict] = {}
        logger.info("Chronos forecasting %d tickers …", len(price_arrays))

        for ticker, prices in price_arrays.items():
            chronos_pred = self._forecast_ticker(ticker, prices)
            tfm_pred = timesfm_predictions.get(ticker, {})
            agreement, warning = self._compare(chronos_pred, tfm_pred)

            results[ticker] = {
                **chronos_pred,
                "agreement_score":    round(agreement, 4),
                "divergence_warning": warning,
            }

        logger.info("Chronos forecasting complete")
        return results

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_pipeline_loaded(self):
        if self._pipeline is not None:
            return
        try:
            import torch
            from chronos import ChronosBoltPipeline

            # Resolve compute device: GPU first, CPU as fallback
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                gpu_name = torch.cuda.get_device_name(0)
                device = "cuda"
                logger.info("Chronos: GPU detected — using CUDA (%s)", gpu_name)
            else:
                device = "cpu"
                logger.info("Chronos: No GPU detected — using CPU (inference will be slower)")

            logger.info("Loading Chronos model %s on %s …", self.model_id, device)
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_id,
                device_map=device,
                dtype=torch.bfloat16,
                cache_dir="./models/cache",
            )
            logger.info("Chronos Bolt model loaded on %s", device)
        except Exception as exc:
            logger.critical(
                "FATAL — Chronos model could not be loaded: %s\n"
                "  Check that 'chronos-forecasting' is installed:  pip install chronos-forecasting\n"
                "  Check internet / HuggingFace access for: %s\n"
                "  Pipeline is stopping.",
                exc, self.model_id,
            )
            raise ModelLoadError(
                f"Chronos failed to load ({self.model_id}): {exc}"
            ) from exc

    # ── Forecasting ───────────────────────────────────────────────────────────

    def _forecast_ticker(self, ticker: str, prices: np.ndarray) -> Dict[str, dict]:
        if len(prices) < _MIN_CONTEXT:
            raise ValueError(
                f"{ticker} has only {len(prices)} data points "
                f"(minimum required: {_MIN_CONTEXT})"
            )

        import torch

        context = torch.tensor(
            prices[-min(512, len(prices)):].astype(np.float32),
            dtype=torch.float32,
        ).unsqueeze(0)   # shape (1, context_len)

        last_price = float(prices[-1])
        out = {}

        for name, horizon in self.horizons.items():
            # Chronos Bolt supports up to 64 steps natively; cap there.
            pred_len = min(horizon, 64)

            # Returns (batch=1, n_quantiles=9, pred_len)
            # Quantile order: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            forecast = self._pipeline.predict(
                context,
                prediction_length=pred_len,
            )

            quantiles = forecast[0].float().numpy()   # (9, pred_len)
            point = float(quantiles[4, -1])   # median (q50)
            low   = float(quantiles[0, -1])   # q10
            high  = float(quantiles[8, -1])   # q90

            out[name] = {
                "point":      round(point, 4),
                "low":        round(low, 4),
                "high":       round(high, 4),
                "pct_change": round((point - last_price) / last_price * 100, 2),
            }

        return out

    # ── Agreement / divergence ────────────────────────────────────────────────

    def _compare(
        self,
        chronos: Dict[str, dict],
        timesfm: Dict[str, dict],
    ):
        """
        agreement_score: fraction of horizons where both models agree on
        direction (both bullish or both bearish).
        divergence_warning: True if any horizon magnitude differs by more
        than _DIVERGENCE_THRESHOLD.
        """
        if not timesfm:
            return 0.5, False

        agreements = []
        diverged   = False

        for name in self.horizons:
            c = chronos.get(name, {})
            t = timesfm.get(name, {})
            if not c or not t:
                continue

            c_dir = 1 if c.get("pct_change", 0) >= 0 else -1
            t_dir = 1 if t.get("pct_change", 0) >= 0 else -1
            agreements.append(1.0 if c_dir == t_dir else 0.0)

            # Relative magnitude divergence
            c_pt = c.get("point", 0)
            t_pt = t.get("point", 0)
            if t_pt and t_pt != 0:
                rel_diff = abs(c_pt - t_pt) / abs(t_pt)
                if rel_diff > _DIVERGENCE_THRESHOLD:
                    diverged = True

        score = float(np.mean(agreements)) if agreements else 0.5
        return score, diverged

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_str
