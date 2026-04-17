"""
ML Meta-Model
─────────────
Trains one gradient-boosted regressor per forecast horizon on the pooled
cross-sectional history of all tickers.

Design
  • Features  : 18 technical + market features (no look-ahead)
  • Label     : forward N-day percentage return (clipped ±80 %)
  • Training  : XGBoost (sklearn GradientBoosting as fallback)
  • Confidence: sigmoid of |predicted_return| — high-magnitude predictions
                are treated as higher-confidence signals

The models are trained purely on past features; TimesFM/Chronos predictions
are combined with the ML prediction downstream in FusionAgent (0.4/0.3/0.3).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("ml_model")

# Features used for training and inference (must be produced by build_feature_frame
# or the market-feature merge in FeatureAgent)
_FEATURE_COLS: List[str] = [
    "rsi_14",
    "macd_hist",
    "bb_pct",
    "volatility_5d",
    "volatility_21d",
    "volatility_63d",
    "momentum_21d",
    "momentum_63d",
    "golden_cross",
    "return_1d",
    "return_5d",
    "return_21d",
    "sma_ratio_10_50",
    "sma_ratio_50_200",
    "drawdown_21d",
    "trend_strength",
    "spy_return_21d",
    "vix_level",
]

_HORIZON_DAYS: Dict[str, int] = {"short": 21, "medium": 126, "long": 252}
_MIN_TRAIN_ROWS = 60   # minimum total rows to attempt training


class MLModel:
    """Train one GBM per horizon; predict latest feature row per ticker."""

    def __init__(self, config: dict):
        cfg = config.get("ml_model", {})
        self.n_estimators  = int(cfg.get("n_estimators",  200))
        self.max_depth     = int(cfg.get("max_depth",      4))
        self.learning_rate = float(cfg.get("learning_rate", 0.05))
        self.models: Dict[str, object] = {}
        self._trained = False

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, featured_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Build pooled supervised datasets from featured_data history and fit
        one model per horizon. Returns True if at least one model trained.
        """
        logger.info("ML meta-model: building training datasets …")
        datasets = self._build_datasets(featured_data)

        for horizon, (X, y) in datasets.items():
            if len(X) < _MIN_TRAIN_ROWS:
                logger.warning(
                    "ML %s: only %d rows — skipping (need ≥%d)",
                    horizon, len(X), _MIN_TRAIN_ROWS,
                )
                continue
            try:
                model = self._fit(X, y)
                self.models[horizon] = model
                logger.info(
                    "ML %s model trained on %d samples × %d features",
                    horizon, len(X), X.shape[1],
                )
            except Exception as exc:
                logger.error("ML training failed for %s: %s", horizon, exc)

        self._trained = bool(self.models)
        return self._trained

    def predict(
        self, featured_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, dict]]:
        """
        For each ticker, predict from its latest feature row.

        Returns
        -------
        {
            ticker: {
                "short":  {"predicted_return": float, "confidence": float},
                "medium": {...},
                "long":   {...},
            }
        }
        """
        if not self._trained:
            logger.warning("ML model not trained — returning neutral predictions")
            return self._empty_predictions(list(featured_data.keys()))

        results: Dict[str, Dict[str, dict]] = {}

        for ticker, df in featured_data.items():
            feat_vec = self._latest_feature_vec(df)
            X = np.array(feat_vec, dtype=np.float32).reshape(1, -1)
            ticker_preds: Dict[str, dict] = {}

            for horizon, model in self.models.items():
                try:
                    pred = float(model.predict(X)[0])
                    conf = self._confidence(pred)
                    ticker_preds[horizon] = {
                        "predicted_return": round(pred, 6),
                        "confidence":       round(conf, 4),
                    }
                except Exception as exc:
                    logger.debug("ML predict failed %s/%s: %s", ticker, horizon, exc)
                    ticker_preds[horizon] = {"predicted_return": 0.0, "confidence": 0.5}

            # Fill any missing horizons (e.g. if training failed for one)
            for h in _HORIZON_DAYS:
                if h not in ticker_preds:
                    ticker_preds[h] = {"predicted_return": 0.0, "confidence": 0.5}

            results[ticker] = ticker_preds

        return results

    # ── Dataset construction ──────────────────────────────────────────────────

    def _build_datasets(
        self, featured_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Pool all tickers into one (X, y) pair per horizon."""
        all_X: Dict[str, List] = {h: [] for h in _HORIZON_DAYS}
        all_y: Dict[str, List] = {h: [] for h in _HORIZON_DAYS}

        for ticker, df in featured_data.items():
            df = df.copy().sort_index()
            close = df["Close"]

            for horizon, n_days in _HORIZON_DAYS.items():
                # Forward return label — naturally NaN for last n_days rows
                fwd_ret = (close.shift(-n_days) / close - 1).clip(-0.8, 0.8)

                feat_df = self._feature_df(df)
                feat_df["_y"] = fwd_ret

                # Drop rows where label is NaN (i.e. last n_days); keep feature NaNs filled below
                valid = feat_df.dropna(subset=["_y"])
                if len(valid) < 10:
                    continue

                cols = [c for c in _FEATURE_COLS if c in valid.columns]
                if not cols:
                    continue

                X_raw = np.zeros((len(valid), len(_FEATURE_COLS)), dtype=np.float32)
                for j, col in enumerate(_FEATURE_COLS):
                    if col in valid.columns:
                        X_raw[:, j] = valid[col].values.astype(np.float32)

                X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                y_raw = valid["_y"].values.astype(np.float32)

                all_X[horizon].extend(X_raw.tolist())
                all_y[horizon].extend(y_raw.tolist())

        datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for h in _HORIZON_DAYS:
            if all_X[h]:
                datasets[h] = (
                    np.array(all_X[h], dtype=np.float32),
                    np.array(all_y[h], dtype=np.float32),
                )
        return datasets

    def _feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract _FEATURE_COLS from df; fill completely missing cols with 0."""
        out = pd.DataFrame(index=df.index)
        for col in _FEATURE_COLS:
            out[col] = df[col] if col in df.columns else 0.0
        return out

    def _latest_feature_vec(self, df: pd.DataFrame) -> List[float]:
        """Return the most recent row as a list of floats (NaN → 0)."""
        feat = self._feature_df(df)
        last = feat.iloc[-1]
        return [
            float(v) if not (isinstance(v, float) and math.isnan(v)) else 0.0
            for v in last.values
        ]

    # ── Model fitting ─────────────────────────────────────────────────────────

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Try XGBoost; fall back to sklearn GradientBoostingRegressor."""
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X, y)
            logger.debug("XGBoost used for ML meta-model")
            return model
        except ImportError:
            logger.info("XGBoost not installed — using sklearn GradientBoosting")
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 100),
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                min_samples_leaf=5,
            )
            model.fit(X, y)
            return model

    # ── Confidence ────────────────────────────────────────────────────────────

    @staticmethod
    def _confidence(pred_return: float) -> float:
        """
        Sigmoid of |pred_return| * 8.
        A 10 % predicted return → confidence ≈ 0.55.
        A 30 % predicted return → confidence ≈ 0.74.
        A near-zero prediction → confidence ≈ 0.50 (maximum uncertainty).
        """
        return float(1.0 / (1.0 + math.exp(-abs(pred_return) * 8)))

    @staticmethod
    def _empty_predictions(tickers: List[str]) -> Dict[str, Dict[str, dict]]:
        return {
            t: {
                h: {"predicted_return": 0.0, "confidence": 0.5}
                for h in _HORIZON_DAYS
            }
            for t in tickers
        }
