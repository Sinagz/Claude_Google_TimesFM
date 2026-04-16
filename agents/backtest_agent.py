"""
Backtesting Agent
──────────────────
Simulates a simple equal-weight rebalancing strategy:
  • Hold the top-N stocks ranked by technical score (no look-ahead).
  • Rebalance every *rebalance_days* calendar days.
  • Long-only, no transaction costs (conservative simplification).

Metrics computed
  • Cumulative return
  • Annualised return
  • Annualised Sharpe ratio (risk-free rate = 0)
  • Maximum drawdown

All calculations are purely historical — no forecast data is used here,
so there is zero look-ahead bias.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.helpers import setup_logger
from utils.indicators import compute_technical_score, build_feature_frame

logger = setup_logger("backtest_agent")

_TRADING_DAYS_PER_YEAR = 252


class BacktestAgent:
    def __init__(self, config: dict):
        self.lookback_days  = config["backtest"]["lookback_days"]
        self.rebalance_days = config["backtest"]["rebalance_days"]
        self.top_n          = config["ranking"]["top_n"]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, raw_data: Dict[str, pd.DataFrame]) -> dict:
        """
        Parameters
        ----------
        raw_data : {ticker -> OHLCV DataFrame}  (same as DataAgent output)

        Returns
        -------
        {
            "cumulative_return":   float,
            "annualised_return":   float,
            "sharpe_ratio":        float,
            "max_drawdown":        float,
            "n_rebalances":        int,
            "benchmark_return":    float,   # equal-weight buy-and-hold all tickers
            "strategy_vs_bench":   float,   # alpha
            "equity_curve":        [[date, portfolio_value], ...],
        }
        """
        logger.info("Running backtest over %d-day history …", self.lookback_days)

        # Align all price series on a common date range
        prices = self._build_price_matrix(raw_data)
        if prices.empty or len(prices) < 30:
            logger.warning("Not enough data for backtest")
            return self._empty_result()

        # Restrict to look-back window
        prices = prices.iloc[-self.lookback_days :]
        n_days = len(prices)

        # ── Strategy equity curve ─────────────────────────────────────────
        portfolio_values = self._simulate(prices)

        # ── Benchmark: equal-weight buy-and-hold ─────────────────────────
        bench_ret = self._benchmark_return(prices)

        # ── Metrics ───────────────────────────────────────────────────────
        cum_ret    = float(portfolio_values[-1] / portfolio_values[0] - 1)
        ann_ret    = float((1 + cum_ret) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1)
        daily_rets = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe     = self._sharpe(daily_rets)
        max_dd     = self._max_drawdown(portfolio_values)
        win_rate   = self._win_rate(daily_rets)

        # Build equity curve for plotting
        dates = prices.index.strftime("%Y-%m-%d").tolist()
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
            "Backtest result — cum_ret=%.2f%%  sharpe=%.2f  max_dd=%.2f%%  win_rate=%.1f%%",
            cum_ret * 100, sharpe, max_dd * 100, win_rate * 100,
        )
        return result

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Walk forward through *prices* day by day.
        Every *rebalance_days* days, re-rank tickers by trailing 21-day
        technical score and switch holdings.
        """
        tickers   = list(prices.columns)
        n_days    = len(prices)
        portfolio = np.ones(n_days)   # start at 1.0 (normalised)
        holdings  = []                 # list of ticker names currently held

        for i in range(n_days):
            # Rebalance on schedule
            if i % self.rebalance_days == 0:
                available = prices.iloc[: i + 1]
                if len(available) >= 30:
                    holdings = self._select_holdings(available, tickers)
                else:
                    holdings = tickers[: self.top_n]

            if i == 0:
                continue

            # Daily portfolio return = average return of held tickers
            day_rets = []
            for t in holdings:
                prev = prices[t].iloc[i - 1]
                curr = prices[t].iloc[i]
                if prev > 0 and not np.isnan(prev) and not np.isnan(curr):
                    day_rets.append((curr - prev) / prev)

            daily_ret = float(np.mean(day_rets)) if day_rets else 0.0
            portfolio[i] = portfolio[i - 1] * (1 + daily_ret)

        return portfolio

    def _select_holdings(
        self, prices: pd.DataFrame, tickers: List[str]
    ) -> List[str]:
        """Rank tickers by trailing technical score; return top-N."""
        scores = {}
        for t in tickers:
            col = prices[t].dropna()
            if len(col) < 30:
                continue
            mini_df = pd.DataFrame({"Close": col})
            try:
                enriched = build_feature_frame(mini_df)
                scores[t] = compute_technical_score(enriched)
            except Exception:
                scores[t] = 0.5
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return ranked[: self.top_n] or tickers[: self.top_n]

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
        Build a wide DataFrame (dates × tickers) of Close prices.
        Drops tickers with more than 10 % missing data.
        Forward-fills minor gaps.
        """
        frames = {}
        for ticker, df in raw_data.items():
            if "Close" not in df.columns:
                continue
            s = df["Close"].copy()
            # Normalise index to tz-naive date-only so US + Canadian calendars align
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, utc=True)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = s.index.normalize()   # strip intraday time component
            s = s[~s.index.duplicated(keep="last")]
            frames[ticker] = s

        if not frames:
            return pd.DataFrame()

        combined = pd.DataFrame(frames)
        combined = combined.sort_index()

        # Drop tickers with > 10 % missing days (e.g. holidays-only gaps)
        thresh = int(len(combined) * 0.90)
        combined = combined.dropna(thresh=thresh, axis=1)
        combined = combined.ffill().bfill()

        return combined

    # ── Statistics ───────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(daily_rets: np.ndarray) -> float:
        if len(daily_rets) < 2 or daily_rets.std() == 0:
            return 0.0
        return float(
            daily_rets.mean() / daily_rets.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown (negative number → fraction)."""
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / peak
        return float(dd.min())

    @staticmethod
    def _win_rate(daily_rets: np.ndarray) -> float:
        """Fraction of trading days with a positive portfolio return."""
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
