"""
Backtesting Agent
──────────────────
Walk-forward simulation of an equal-weight (or confidence-weighted) top-N
rebalancing strategy on historical price data.

Walk-forward design
  At each rebalance point *i*, only data up to day *i* is used to select
  holdings (no look-ahead).  Holdings are held for the next *rebalance_days*
  period then re-evaluated.

Position sizing (when ml_predictions provided)
  position_weight[t] = confidence[t] / Σ confidence[held tickers]
  Otherwise equal-weight: position_weight[t] = 1 / N

Metrics
  • Cumulative return
  • Annualised return
  • Sharpe ratio  (risk-free rate = 0)
  • Maximum drawdown
  • Win rate  (fraction of positive-return days)
  • Portfolio equity curve
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger
from utils.indicators import compute_technical_score, build_feature_frame

logger = setup_logger("backtest_agent")

_TRADING_DAYS_PER_YEAR = 252


class BacktestAgent:
    def __init__(self, config: dict):
        bt = config["backtest"]
        self.lookback_days  = bt["lookback_days"]
        self.rebalance_days = bt["rebalance_days"]
        self.train_window   = bt.get("train_window", 180)
        self.top_n          = config["ranking"]["top_n"]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        raw_data:       Dict[str, pd.DataFrame],
        ml_predictions: Optional[Dict[str, Dict[str, dict]]] = None,
    ) -> dict:
        """
        Parameters
        ----------
        raw_data       : {ticker -> OHLCV DataFrame}  (same as DataAgent output)
        ml_predictions : {ticker -> {horizon -> {predicted_return, confidence}}}
                         Optional — used for confidence-weighted position sizing.

        Returns
        -------
        {
            "cumulative_return", "annualised_return", "sharpe_ratio",
            "max_drawdown", "win_rate", "n_rebalances",
            "benchmark_return", "strategy_vs_bench", "equity_curve"
        }
        """
        logger.info("Running backtest over %d-day history …", self.lookback_days)

        prices = self._build_price_matrix(raw_data)
        if prices.empty or len(prices) < 30:
            logger.warning("Not enough data for backtest")
            return self._empty_result()

        prices = prices.iloc[-self.lookback_days:]
        n_days = len(prices)

        portfolio_values = self._simulate(prices, ml_predictions)

        bench_ret  = self._benchmark_return(prices)
        cum_ret    = float(portfolio_values[-1] / portfolio_values[0] - 1)
        ann_ret    = float((1 + cum_ret) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1)
        daily_rets = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe     = self._sharpe(daily_rets)
        max_dd     = self._max_drawdown(portfolio_values)
        win_rate   = self._win_rate(daily_rets)

        dates        = prices.index.strftime("%Y-%m-%d").tolist()
        equity_curve = [[d, round(float(v), 4)] for d, v in zip(dates, portfolio_values)]

        result = {
            "cumulative_return":  round(cum_ret,   4),
            "annualised_return":  round(ann_ret,   4),
            "sharpe_ratio":       round(sharpe,    4),
            "max_drawdown":       round(max_dd,    4),
            "win_rate":           round(win_rate,  4),
            "n_rebalances":       max(1, n_days // self.rebalance_days),
            "benchmark_return":   round(bench_ret, 4),
            "strategy_vs_bench":  round(cum_ret - bench_ret, 4),
            "equity_curve":       equity_curve,
        }

        logger.info(
            "Backtest — cum=%.2f%%  sharpe=%.2f  maxDD=%.2f%%  winRate=%.1f%%",
            cum_ret * 100, sharpe, max_dd * 100, win_rate * 100,
        )
        return result

    # ── Walk-forward simulation ───────────────────────────────────────────────

    def _simulate(
        self,
        prices:         pd.DataFrame,
        ml_predictions: Optional[Dict[str, Dict[str, dict]]],
    ) -> np.ndarray:
        """
        Walk forward day by day.
        Every *rebalance_days* days, re-score and rebalance using only data
        up to that point (train window = last *train_window* days).
        Position weights are proportional to ML confidence when available;
        otherwise equal-weight.
        """
        tickers   = list(prices.columns)
        n_days    = len(prices)
        portfolio = np.ones(n_days)
        holdings: List[str]           = []
        weights:  Dict[str, float]    = {}

        for i in range(n_days):
            # Rebalance: walk-forward — use only past data up to day i
            if i % self.rebalance_days == 0:
                train_start = max(0, i - self.train_window)
                avail = prices.iloc[train_start: i + 1]
                if len(avail) >= 30:
                    holdings, weights = self._select_holdings_weighted(
                        avail, tickers, ml_predictions
                    )
                else:
                    holdings = tickers[: self.top_n]
                    n = len(holdings)
                    weights  = {t: 1.0 / n for t in holdings} if n else {}

            if i == 0 or not holdings:
                continue

            total_w  = sum(weights.get(t, 0.0) for t in holdings)
            if total_w == 0.0:
                portfolio[i] = portfolio[i - 1]
                continue

            day_ret = 0.0
            for t in holdings:
                prev = prices[t].iloc[i - 1]
                curr = prices[t].iloc[i]
                w    = weights.get(t, 0.0) / total_w
                if prev > 0 and not np.isnan(prev) and not np.isnan(curr):
                    day_ret += w * (curr - prev) / prev

            portfolio[i] = portfolio[i - 1] * (1 + day_ret)

        return portfolio

    def _select_holdings_weighted(
        self,
        prices:         pd.DataFrame,
        tickers:        List[str],
        ml_predictions: Optional[Dict[str, Dict[str, dict]]],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Rank by technical score; weight by ML confidence (short horizon)."""
        scores:      Dict[str, float] = {}
        confidences: Dict[str, float] = {}

        for t in tickers:
            col = prices[t].dropna()
            if len(col) < 30:
                continue
            mini_df = pd.DataFrame({"Close": col})
            try:
                enriched    = build_feature_frame(mini_df)
                scores[t]   = compute_technical_score(enriched)
            except Exception:
                scores[t]   = 0.5
            if ml_predictions and t in ml_predictions:
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

    # ── Benchmark ────────────────────────────────────────────────────────────

    @staticmethod
    def _benchmark_return(prices: pd.DataFrame) -> float:
        """Equal-weight buy-and-hold cumulative return."""
        rets = []
        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) >= 2:
                rets.append(float(series.iloc[-1] / series.iloc[0] - 1))
        return float(np.mean(rets)) if rets else 0.0

    # ── Price matrix ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_price_matrix(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Wide DataFrame (dates × tickers) of Close prices.
        Normalises to tz-naive date-only per series before joining so that
        mixed US/Canadian trading calendars align correctly.
        """
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
            s = s[~s.index.duplicated(keep="last")]
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
            "sharpe_ratio":       0.0,
            "max_drawdown":       0.0,
            "win_rate":           0.0,
            "n_rebalances":       0,
            "benchmark_return":   0.0,
            "strategy_vs_bench":  0.0,
            "equity_curve":       [],
        }
