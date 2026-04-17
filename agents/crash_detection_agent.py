"""
Crash Detection Agent
──────────────────────
Detects market stress / crash regimes using multiple statistical signals:

  1. VIX z-score           — how elevated volatility is vs. its own history
  2. SPY momentum          — sustained downtrend in the broad market
  3. Correlation spike     — cross-stock correlations surge during crashes
  4. Market breadth        — fraction of stocks with recent negative returns
  5. Vol-of-vol            — second-order volatility (uncertainty spike)

All signals are combined via a sigmoid composite into crash_probability ∈ [0,1].

Regime labels
  bull            : crash_prob < 0.25
  neutral         : 0.25 ≤ crash_prob < 0.50
  high_volatility : 0.50 ≤ crash_prob < 0.75
  crash           : crash_prob ≥ 0.75
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("crash_detection_agent")


def _safe_zscore(series: pd.Series, lookback: int) -> float:
    """Rolling z-score: how many std-devs is latest value vs. past `lookback`."""
    if len(series) < max(10, lookback // 4):
        return 0.0
    recent = series.dropna().iloc[-lookback:]
    if len(recent) < 5:
        return 0.0
    mu  = float(recent.mean())
    std = float(recent.std())
    if std < 1e-8:
        return 0.0
    return float((recent.iloc[-1] - mu) / std)


class CrashDetectionAgent:
    """Produces crash_probability and regime label from market data."""

    def __init__(self, config: dict):
        cfg = config.get("crash_detection", {})
        self.lookback         = int(cfg.get("lookback_days",          60))
        self.vix_high         = float(cfg.get("vix_high",             30.0)) / 100.0
        self.vix_extreme      = float(cfg.get("vix_extreme",          40.0)) / 100.0
        self.corr_threshold   = float(cfg.get("correlation_threshold", 0.75))
        self.breadth_threshold= float(cfg.get("breadth_threshold",    0.65))
        # Signal weights (need not sum to 1 — normalised internally)
        self._w = {
            "vix_zscore":   0.30,
            "spy_momentum": 0.25,
            "correlation":  0.20,
            "breadth":      0.15,
            "vol_of_vol":   0.10,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        featured_data: Dict[str, pd.DataFrame],
        raw_data:      Optional[Dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Parameters
        ----------
        featured_data : {ticker → enriched DataFrame} from FeatureAgent
                        Must have columns: spy_return_21d, vix_level,
                        return_1d, return_5d, volatility_21d (at minimum)
        raw_data      : {ticker → OHLCV} used for breadth / correlation calc

        Returns
        -------
        {
            "crash_probability": float [0, 1],
            "regime":            str,            # bull / neutral / high_volatility / crash
            "signals":           dict,           # per-signal raw values
            "recommendation":    str,
        }
        """
        signals = self._compute_signals(featured_data, raw_data)
        crash_prob = self._combine_signals(signals)
        regime     = self._label_regime(crash_prob)
        recommendation = self._recommend(regime)

        logger.info(
            "Crash detection — prob=%.3f  regime=%s  signals=%s",
            crash_prob, regime,
            {k: f"{v:.3f}" for k, v in signals.items()},
        )
        return {
            "crash_probability": round(float(crash_prob), 4),
            "regime":            regime,
            "signals":           {k: round(float(v), 4) for k, v in signals.items()},
            "recommendation":    recommendation,
        }

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_signals(
        self,
        featured_data: Dict[str, pd.DataFrame],
        raw_data:      Optional[Dict[str, pd.DataFrame]],
    ) -> Dict[str, float]:
        vix_z   = self._signal_vix_zscore(featured_data)
        spy_mom = self._signal_spy_momentum(featured_data)
        corr    = self._signal_correlation_spike(raw_data or {})
        breadth = self._signal_breadth(featured_data)
        vov     = self._signal_vol_of_vol(featured_data)
        return {
            "vix_zscore":   vix_z,
            "spy_momentum": spy_mom,
            "correlation":  corr,
            "breadth":      breadth,
            "vol_of_vol":   vov,
        }

    def _signal_vix_zscore(self, featured_data: Dict[str, pd.DataFrame]) -> float:
        """Returns [0,1]: 0 = calm VIX, 1 = extremely elevated VIX."""
        vix_series: Optional[pd.Series] = None
        for df in featured_data.values():
            if df is not None and "vix_level" in df.columns:
                s = df["vix_level"].dropna()
                if len(s) > 20:
                    vix_series = s
                    break

        if vix_series is None:
            return 0.2  # neutral assumption

        current_vix = float(vix_series.iloc[-1])

        # Absolute level component
        level_score = np.clip(
            (current_vix - self.vix_high) / (self.vix_extreme - self.vix_high),
            0.0, 1.0,
        )
        # z-score component
        z = _safe_zscore(vix_series, self.lookback)
        z_score = float(np.clip((z + 2.0) / 5.0, 0.0, 1.0))  # z∈[-2,3] → [0,1]

        return float(0.6 * level_score + 0.4 * z_score)

    def _signal_spy_momentum(self, featured_data: Dict[str, pd.DataFrame]) -> float:
        """Returns [0,1]: 0 = bullish SPY, 1 = deep downtrend."""
        spy_21d: Optional[float] = None
        spy_5d:  Optional[float] = None

        for df in featured_data.values():
            if df is None or df.empty:
                continue
            if "spy_return_21d" in df.columns:
                s = df["spy_return_21d"].dropna()
                if not s.empty:
                    spy_21d = float(s.iloc[-1])
            if "return_5d" in df.columns:
                s = df["return_5d"].dropna()
                if not s.empty:
                    spy_5d = float(s.iloc[-1])
            if spy_21d is not None:
                break

        if spy_21d is None:
            return 0.2

        # Map 21d return to stress score: −10% → 1.0, +5% → 0.0
        score_21d = float(np.clip((-spy_21d + 0.05) / 0.15, 0.0, 1.0))
        score_5d  = 0.0
        if spy_5d is not None:
            score_5d = float(np.clip((-spy_5d + 0.02) / 0.08, 0.0, 1.0))

        return float(0.7 * score_21d + 0.3 * score_5d)

    def _signal_correlation_spike(
        self, raw_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Returns [0,1]: average pairwise return correlation among stocks."""
        if not raw_data:
            return 0.2

        ret_series = {}
        for t, df in raw_data.items():
            if df is None or "Close" not in df.columns:
                continue
            closes = df["Close"].dropna().iloc[-self.lookback:]
            if len(closes) > 10:
                ret_series[t] = closes.pct_change().dropna()

        if len(ret_series) < 4:
            return 0.2

        # Sample up to 30 tickers to keep computation fast
        keys = list(ret_series.keys())[:30]
        rets_df = pd.DataFrame({k: ret_series[k] for k in keys}).dropna()
        if rets_df.shape[0] < 5 or rets_df.shape[1] < 2:
            return 0.2

        corr_mat = rets_df.corr().values
        n = corr_mat.shape[0]
        # Mean of upper triangle (excluding diagonal)
        upper = corr_mat[np.triu_indices(n, k=1)]
        mean_corr = float(np.nanmean(upper)) if len(upper) > 0 else 0.0

        # Map: normal ~0.3, crash ~0.8+
        score = float(np.clip((mean_corr - 0.30) / 0.50, 0.0, 1.0))
        return score

    def _signal_breadth(self, featured_data: Dict[str, pd.DataFrame]) -> float:
        """Returns [0,1]: fraction of stocks in recent downtrend (5d return < 0)."""
        negative = 0
        total    = 0
        for df in featured_data.values():
            if df is None or "return_5d" not in df.columns:
                continue
            s = df["return_5d"].dropna()
            if s.empty:
                continue
            total += 1
            if float(s.iloc[-1]) < 0:
                negative += 1

        if total == 0:
            return 0.2
        breadth_down = negative / total
        # Normal ~30%, crash ~80%+
        return float(np.clip((breadth_down - 0.30) / 0.50, 0.0, 1.0))

    def _signal_vol_of_vol(self, featured_data: Dict[str, pd.DataFrame]) -> float:
        """Returns [0,1]: standard deviation of 21-day volatility (instability)."""
        vol_latest = []
        for df in featured_data.values():
            if df is None or "volatility_21d" not in df.columns:
                continue
            s = df["volatility_21d"].dropna()
            if len(s) > 20:
                vol_latest.append(float(s.iloc[-1]))

        if not vol_latest:
            return 0.2

        vov = float(np.std(vol_latest))
        # Map: stable ~0.05, chaotic ~0.20+
        return float(np.clip((vov - 0.05) / 0.15, 0.0, 1.0))

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Weighted average of signals, then logistic compression."""
        total_w = sum(self._w.values())
        raw = sum(
            self._w.get(k, 0.0) * v for k, v in signals.items()
        ) / total_w

        # Mild logistic sharpening so near-zero stays near-zero
        # and near-one stays near-one
        sharpened = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
        return float(np.clip(sharpened, 0.0, 1.0))

    @staticmethod
    def _label_regime(crash_prob: float) -> str:
        if crash_prob >= 0.75:
            return "crash"
        if crash_prob >= 0.50:
            return "high_volatility"
        if crash_prob >= 0.25:
            return "neutral"
        return "bull"

    @staticmethod
    def _recommend(regime: str) -> str:
        return {
            "bull":            "BUY",
            "neutral":         "HOLD",
            "high_volatility": "REDUCE",
            "crash":           "STOP",
        }.get(regime, "HOLD")
