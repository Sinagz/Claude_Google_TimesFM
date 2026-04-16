"""
TimesFM Forecast Agent
───────────────────────
Loads Google's TimesFM 2.5 (PyTorch backend) and generates price forecasts
for three horizons: short, medium, long.

Model source: https://huggingface.co/google/timesfm-2.5-200m-pytorch
Package:      pip install --no-deps git+https://github.com/google-research/timesfm.git

API (timesfm 2.0)
  model = TimesFM_2p5_200M_torch.from_pretrained(repo_id)
  model.compile(ForecastConfig(max_context=..., max_horizon=..., per_core_batch_size=...))
  point, quantile = model.forecast(horizon, [np.array(...)])

HARD REQUIREMENT — no fallback
  TimesFM is a core model. If it cannot load, ModelLoadError is raised and
  the entire pipeline is aborted.
"""

from typing import Dict

import numpy as np

from utils.helpers import setup_logger

logger = setup_logger("timesfm_agent")

_MIN_CONTEXT = 64


class ModelLoadError(RuntimeError):
    """Raised when a required forecasting model fails to load."""


class TimesFMAgent:
    def __init__(self, config: dict):
        self.config = config
        self.model_id = config["model"]["timesfm_checkpoint"]
        self.context_len = config["model"].get("timesfm_context_len", 512)
        self.horizon_len = config["model"].get("timesfm_horizon_len", 256)
        self.horizons = config["forecast"]["horizons"]   # {short, medium, long}
        self.device = self._resolve_device(config["model"]["device"])
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self, price_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, dict]]:
        """
        Raises ModelLoadError if TimesFM cannot be loaded (aborts pipeline).
        Returns {ticker: {short/medium/long: {point, low, high, pct_change}}}
        """
        self._ensure_model_loaded()

        results: Dict[str, Dict[str, dict]] = {}
        logger.info("TimesFM forecasting %d tickers …", len(price_arrays))

        for ticker, prices in price_arrays.items():
            results[ticker] = self._forecast_ticker(ticker, prices)

        logger.info("TimesFM forecasting complete")
        return results

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        try:
            from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
            from huggingface_hub import hf_hub_download
            import torch

            # Resolve compute device: GPU first, CPU as fallback
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("TimesFM: GPU detected — using CUDA (%s)", gpu_name)
            else:
                logger.info("TimesFM: No GPU detected — using CPU (inference will be slower)")

            logger.info("Loading TimesFM model %s …", self.model_id)

            # Download weights manually — avoids the huggingface_hub >= 0.30
            # proxies kwarg bug that breaks TimesFM's PyTorchModelHubMixin path.
            weights_path = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors",
                cache_dir="./models/cache",
            )

            # torch_compile only works reliably on CUDA; skip on CPU
            use_compile = cuda_ok
            self._model = TimesFM_2p5_200M_torch(torch_compile=use_compile)
            self._model.model.load_checkpoint(
                weights_path, torch_compile=use_compile
            )

            # max_horizon must be a multiple of 128 (output patch size = 128)
            # max_context must be a multiple of 32 (input patch size = 32)
            max_h   = self._next_multiple(self.horizon_len, 128)
            max_ctx = self._next_multiple(self.context_len, 32)

            self._model.compile(
                ForecastConfig(
                    max_context=max_ctx,
                    max_horizon=max_h,
                    per_core_batch_size=8,
                    normalize_inputs=True,
                    infer_is_positive=True,
                )
            )

            logger.info(
                "TimesFM loaded (max_context=%d, max_horizon=%d, device=%s)",
                max_ctx, max_h, self.device,
            )
        except Exception as exc:
            logger.critical(
                "FATAL — TimesFM model could not be loaded: %s\n"
                "  Ensure 'timesfm' is installed from GitHub:\n"
                "    pip install --no-deps git+https://github.com/google-research/timesfm.git\n"
                "  And that the HuggingFace checkpoint is reachable: %s\n"
                "  Pipeline is stopping.",
                exc, self.model_id,
            )
            raise ModelLoadError(
                f"TimesFM failed to load ({self.model_id}): {exc}"
            ) from exc

    # ── Forecasting ───────────────────────────────────────────────────────────

    def _forecast_ticker(self, ticker: str, prices: np.ndarray) -> Dict[str, dict]:
        if len(prices) < _MIN_CONTEXT:
            raise ValueError(
                f"{ticker} has only {len(prices)} data points "
                f"(minimum required: {_MIN_CONTEXT})"
            )

        # Use most recent context_len points
        context = prices[-self.context_len :].astype(np.float64)
        last_price = float(context[-1])

        out = {}
        for name, h in self.horizons.items():
            # forecast() returns arrays of shape (n_series, horizon)
            point_arr, quantile_arr = self._model.forecast(h, [context])
            # point_arr: (1, h)  quantile_arr: (1, h, n_quantiles)

            pt = float(point_arr[0, -1])

            if quantile_arr is not None and quantile_arr.ndim == 3 and quantile_arr.shape[2] >= 2:
                q_lo = float(quantile_arr[0, -1, 0])
                q_hi = float(quantile_arr[0, -1, -1])
            else:
                spread = abs(pt - last_price) * 0.3 + last_price * 0.02
                q_lo, q_hi = pt - spread, pt + spread

            out[name] = {
                "point":      round(pt, 4),
                "low":        round(q_lo, 4),
                "high":       round(q_hi, 4),
                "pct_change": round((pt - last_price) / last_price * 100, 2),
            }

        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _next_multiple(n: int, base: int) -> int:
        """Round n up to the next multiple of base."""
        import math
        return math.ceil(n / base) * base

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_str
