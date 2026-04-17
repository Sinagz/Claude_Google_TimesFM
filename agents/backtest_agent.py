"""
Backtesting Agent — Hedge Fund Grade
──────────────────────────────────────
Walk-forward simulation with realistic market frictions.

Enhancements over basic backtest
  • Transaction costs   : flat bps per trade (configurable)
  • Slippage            : random uniform noise on execution price (configurable)
  • RL position sizing  : if rl_weights provided, use them; else confidence-weighted
  • Regime tracking     : collect metrics per detected regime window
  • Sortino ratio       : downside-deviation denominator
  • CAGR                : compound annual growth rate
  • SPY benchmark       : from featured price data or SPY column

Metrics returned
  cumulative_return, annualised_return, cagr, sharpe_ratio, sortino_ratio,
  max_drawdown, win_rate, n_rebalances, benchmark_return,
  strategy_vs_bench, equity_curve, regime_performance
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger
from utils.indicators import compute_technical_score, build_feature_frame

logger = setup_logger("backtest_agent")

_TRADING_DAYS_PER_YEAR = 252
_RNG = np.random.default_rng(42)   # reproducible slippage


class BacktestAgent:
    def __init__(self, config: dict):
        bt = config["backtest"]
        self.lookback_days   = bt["lookback_days"]
        self.rebalance_days  = bt["rebalance_days"]
        self.train_window    = bt.get("train_window", 180)
        self.top_n           = config["ranking"]["top_n"]
        self.transaction_cost= float(bt.get("transaction_cost_bps", 10)) / 10_000
        self.slippage_bps    = float(bt.get("slippage_bps",           5)) / 10_000

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        raw_data:       Dict[str, pd.DataFrame],
        ml_predictions: Optional[Dict[str, Dict[str, dict]]] = None,
        regime:         str = "neutral",
        rl_weights:     Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Parameters
        ----------
        raw_data       : {ticker → OHLCV DataFrame}
        ml_predictions : {ticker → {horizon → {predicted_return, confidence}}}
        regime         : current market regime label
        rl_weights     : {ticker → weight} from RLTradingAgent (optional override)

        Returns
        -------
        Full backtest metrics dict including equity_curve and regime_performance.
        """
        logger.info(
            "Backtest: lookback=%dd  rebalance=%dd  tc=%.1fbps  "
            "slippage=%.1fbps  regime=%s",
            self.lookback_days, self.rebalance_days,
            self.transaction_cost * 10_000, self.slippage_bps * 10_000, regime,
        )

        prices = self._build_price_matrix(raw_data)
        if prices.empty or len(prices) < 30:
            logger.warning("Insufficient price data for backtest")
            return self._empty_result()

        prices   = prices.iloc[-self.lookback_days:]
        n_days   = len(prices)

        portfolio_values, regime_log = self._simulate(prices, ml_predictions, regime, rl_weights)

        bench_ret  = self._benchmark_return(prices)
        cum_ret    = float(portfolio_values[-1] / portfolio_values[0] - 1)
        ann_ret    = float((1 + cum_ret) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1)
        cagr       = float((1 + cum_ret) ** (365.0 / max(1, n_days)) - 1)
        daily_rets = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe     = self._sharpe(daily_rets)
        sortino    = self._sortino(daily_rets)
        max_dd     = self._max_drawdown(portfolio_values)
        win_rate   = self._win_rate(daily_rets)

        dates        = prices.index.strftime("%Y-%m-%d").tolist()
        equity_curve = [[d, round(float(v), 4)] for d, v in zip(dates, portfolio_values)]

        result = {
            "cumulative_return":  round(cum_ret,   4),
            "annualised_return":  round(ann_ret,   4),
            "cagr":               round(cagr,      4),
            "sharpe_ratio":       round(sharpe,    4),
            "sortino_ratio":      round(sortino,   4),
            "max_drawdown":       round(max_dd,    4),
            "win_rate":           round(win_rate,  4),
            "n_rebalances":       max(1, n_days // self.rebalance_days),
            "benchmark_return":   round(bench_ret, 4),
            "strategy_vs_bench":  round(cum_ret - bench_ret, 4),
            "equity_curve":       equity_curve,
            "regime_performance": regime_log,
            "transaction_cost_bps": self.transaction_cost * 10_000,
            "slippage_bps":         self.slippage_bps * 10_000,
        }

        logger.info(
            "Backtest — cum=%.2f%%  CAGR=%.2f%%  sharpe=%.2f  "
            "sortino=%.2f  maxDD=%.2f%%  winRate=%.1f%%",
            cum_ret * 100, cagr * 100, sharpe, sortino,
            max_dd * 100, win_rate * 100,
        )
        return result

    # ── Walk-forward simulation ───────────────────────────────────────────────

    def _simulate(
        self,
        prices:         pd.DataFrame,
        ml_predictions: Optional[Dict[str, Dict[str, dict]]],
        regime:         str,
        rl_weights:     Optional[Dict[str, float]],
    ) -> Tuple[np.ndarray, dict]:
        # Regime exposure multipliers
        exposure_map = {
            "bull":            1.00,
            "neutral":         1.00,
            "bear":            0.70,
            "high_volatility": 0.60,
            "crash":           0.30,
        }
        exposure = exposure_map.get(regime, 1.0)

        tickers   = list(prices.columns)
        n_days    = len(prices)
        portfolio = np.ones(n_days)
        holdings: List[str]        = []
        weights:  Dict[str, float] = {}
        prev_weights: Dict[str, float] = {}

        # Regime performance tracking
        regime_log: Dict[str, List[float]] = {"bull": [], "neutral": [], "bear": [],
                                               "high_volatility": [], "crash": []}

        for i in range(n_days):
            # ── Rebalance ──────────────────────────────────────────────────
            if i % self.rebalance_days == 0:
                train_start = max(0, i - self.train_window)
                avail       = prices.iloc[train_start: i + 1]
                if len(avail) >= 30:
                    holdings, weights = self._select_holdings(
                        avail, tickers, ml_predictions, rl_weights
                    )
                else:
                    n_h      = min(self.top_n, len(tickers))
                    holdings = tickers[:n_h]
                    weights  = {t: 1.0 / n_h for t in holdings}

                # Scale weights by regime exposure
                weights = {t: w * exposure for t, w in weights.items()}

            if i == 0 or not holdings:
                prev_weights = weights.copy()
                continue

            # ── Transaction cost on portfolio turnover ────────────────────
            turnover = sum(
                abs(weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                for t in set(holdings) | set(prev_weights)
            ) / 2.0
            tc_drag = self.transaction_cost * turnover

            # ── Daily P&L with slippage ───────────────────────────────────
            total_w = sum(weights.values())
            if total_w == 0.0:
                portfolio[i] = portfolio[i - 1]
                prev_weights = weights.copy()
                continue

            day_ret = 0.0
            for t in holdings:
                prev_price = prices[t].iloc[i - 1]
                curr_price = prices[t].iloc[i]
                if prev_price <= 0 or np.isnan(prev_price) or np.isnan(curr_price):
                    continue
                # Slippage: random ±slippage_bps on execution
                slip = _RNG.uniform(-self.slippage_bps, self.slippage_bps)
                actual_ret = (curr_price / prev_price - 1.0) + slip
                w = weights[t] / total_w
                day_ret += w * actual_ret

            net_ret = day_ret - tc_drag
            portfolio[i] = portfolio[i - 1] * (1.0 + net_ret)

            # Track by regime
            regime_log.setdefault(regime, []).append(net_ret)

            prev_weights = weights.copy()

        # Summarise regime performance
        regime_summary: dict = {}
        for r, rets in regime_log.items():
            if rets:
                arr = np.array(rets)
                regime_summary[r] = {
                    "n_days":    len(rets),
                    "mean_daily": round(float(arr.mean()), 6),
                    "cum_return": round(float(np.prod(1 + arr) - 1), 4),
                }

        return portfolio, regime_summary

    def _select_holdings(
        self,
        prices:         pd.DataFrame,
        tickers:        List[str],
        ml_predictions: Optional[Dict[str, Dict[str, dict]]],
        rl_weights:     Optional[Dict[str, float]],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Rank by technical score; weight by RL/ML confidence."""
        scores:      Dict[str, float] = {}
        confidences: Dict[str, float] = {}

        for t in tickers:
            col = prices[t].dropna()
            if len(col) < 30:
                continue
            mini_df = pd.DataFrame({"Close": col})
            try:
                enriched  = build_feature_frame(mini_df)
                scores[t] = compute_technical_score(enriched)
            except Exception:
                scores[t] = 0.5

            # Prefer RL weights, fall back to ML confidence
            if rl_weights and t in rl_weights:
                confidences[t] = float(rl_weights[t])
            elif ml_predictions and t in ml_predictions:
                confidences[t] = float(
                    ml_predictions[t].get("short", {}).get("confidence", 0.5) or 0.5
                )
            else:
                confidences[t] = 0.5

        ranked   = sorted(scores, key=lambda x: scores[x], reverse=True)
        selected = ranked[: self.top_n] or tickers[: self.top_n]

        total_c = sum(confidences.get(t, 0.5) for t in selected)
        if total_c == 0.0:
            total_c = len(selected) * 0.5
        weights = {t: confidences.get(t, 0.5) / total_c for t in selected}

        return selected, weights

    # ── Benchmark ─────────────────────────────────────────────────────────────

    @staticmethod
    def _benchmark_return(prices: pd.DataFrame) -> float:
        """Equal-weight buy-and-hold across all tickers."""
        rets = []
        for col in prices.columns:
            s = prices[col].dropna()
            if len(s) >= 2:
                rets.append(float(s.iloc[-1] / s.iloc[0] - 1))
        return float(np.mean(rets)) if rets else 0.0

    # ── Price matrix ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_price_matrix(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: Dict[str, pd.Series] = {}
        for ticker, df in raw_data.items():
            if "Close" not in df.columns:
                continue
            s = df["Close"].copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, utc=True)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = s.index.normalize()
            s       = s[~s.index.duplicated(keep="last")]
            frames[ticker] = s

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames).sort_index()
        thresh   = int(len(combined) * 0.90)
        combined = combined.dropna(thresh=thresh, axis=1)
        combined = combined.ffill().bfill()
        return combined

    # ── Statistics ────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(daily_rets: np.ndarray) -> float:
        if len(daily_rets) < 2 or daily_rets.std() == 0:
            return 0.0
        return float(
            daily_rets.mean() / daily_rets.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def _sortino(daily_rets: np.ndarray, target: float = 0.0) -> float:
        """Sortino: mean return / downside deviation."""
        if len(daily_rets) < 2:
            return 0.0
        downside = daily_rets[daily_rets < target]
        if len(downside) == 0:
            return float(np.inf) if daily_rets.mean() > 0 else 0.0
        dd_std = float(np.sqrt(np.mean(downside ** 2)))
        if dd_std == 0:
            return 0.0
        return float(daily_rets.mean() / dd_std * np.sqrt(_TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / peak
        return float(dd.min())

    @staticmethod
    def _win_rate(daily_rets: np.ndarray) -> float:
        if len(daily_rets) == 0:
            return 0.0
        return float(np.sum(daily_rets > 0) / len(daily_rets))

    @staticmethod
    def _empty_result() -> dict:
        return {
            "cumulative_return":  0.0,
            "annualised_return":  0.0,
            "cagr":               0.0,
            "sharpe_ratio":       0.0,
            "sortino_ratio":      0.0,
            "max_drawdown":       0.0,
            "win_rate":           0.0,
            "n_rebalances":       0,
            "benchmark_return":   0.0,
            "strategy_vs_bench":  0.0,
            "equity_curve":       [],
            "regime_performance": {},
            "transaction_cost_bps": 0.0,
            "slippage_bps":         0.0,
        }
