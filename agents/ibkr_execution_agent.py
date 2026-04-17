"""
IBKR Execution Agent
─────────────────────
Submits orders to Interactive Brokers (paper or live) based on signals
from the RL agent and ranking agent.

Safety rules (HARD-CODED — cannot be overridden at runtime):
  1. Paper trading is the DEFAULT; live requires explicit config flag.
  2. Kill-switch: if crash_probability ≥ kill_switch_threshold → no trades.
  3. No single position may exceed max_position_pct of portfolio.
  4. Total equity exposure capped at max_total_exposure.

Dependencies
  • ib_insync  (pip install ib_insync)
  • Running TWS or IB Gateway:
      Paper: host=127.0.0.1, port=7497
      Live:  host=127.0.0.1, port=7496

When ib_insync is not installed or the connection fails, the agent operates
in DRY-RUN mode and logs what it would have traded without executing anything.
"""

import datetime
import os
from typing import Dict, List, Optional

from utils.helpers import setup_logger

logger = setup_logger("ibkr_execution_agent")

# ── Optional ib_insync import ─────────────────────────────────────────────────
try:
    from ib_insync import IB, LimitOrder, MarketOrder, Stock, util
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False
    logger.warning(
        "ib_insync not installed — IBKR agent will run in DRY-RUN mode. "
        "Install with: pip install ib_insync"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Order record (returned in executed_trades list)
# ─────────────────────────────────────────────────────────────────────────────

class _TradeRecord:
    __slots__ = ("ticker", "action", "quantity", "order_type", "limit_price",
                 "status", "timestamp", "note")

    def __init__(self, ticker, action, quantity, order_type="MARKET",
                 limit_price=None, status="PENDING", note=""):
        self.ticker      = ticker
        self.action      = action       # "BUY" | "SELL"
        self.quantity    = quantity
        self.order_type  = order_type   # "MARKET" | "LIMIT"
        self.limit_price = limit_price
        self.status      = status       # "PENDING" | "FILLED" | "REJECTED" | "DRY_RUN"
        self.timestamp   = datetime.datetime.now().isoformat()
        self.note        = note

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

class IBKRExecutionAgent:
    """
    Converts RL/ranking signals into brokerage orders.

    Parameters (via config["ibkr"])
    ──────────────────────────────
    enabled               : bool  — master on/off switch (default False)
    live_trading          : bool  — MUST be True for live orders (default False)
    host                  : str   — TWS host (default "127.0.0.1")
    paper_port            : int   — TWS paper port (default 7497)
    live_port             : int   — TWS live port  (default 7496)
    client_id             : int   — IB client id (default 1)
    kill_switch_threshold : float — crash_prob ≥ this → no trades (default 0.65)
    max_position_pct      : float — max % of portfolio per ticker (default 0.15)
    max_total_exposure    : float — max total equity exposure (default 0.95)
    order_type            : str   — "MARKET" or "LIMIT" (default "LIMIT")
    limit_slippage_pct    : float — limit price = last_price × (1 ± slippage) (default 0.002)
    portfolio_value       : float — assumed portfolio value in USD (default 100_000)
    dry_run               : bool  — log-only even if connected (default True)
    """

    def __init__(self, config: dict):
        cfg  = config.get("ibkr", {})
        self.enabled           = bool(cfg.get("enabled",               False))
        self.live_trading      = bool(cfg.get("live_trading",          False))  # HARD DEFAULT = FALSE
        self.host              = str (cfg.get("host",                  "127.0.0.1"))
        self.paper_port        = int (cfg.get("paper_port",            7497))
        self.live_port         = int (cfg.get("live_port",             7496))
        self.client_id         = int (cfg.get("client_id",             1))
        self.kill_threshold    = float(cfg.get("kill_switch_threshold", 0.65))
        self.max_position_pct  = float(cfg.get("max_position_pct",    0.15))
        self.max_exposure      = float(cfg.get("max_total_exposure",   0.95))
        self.order_type        = str(cfg.get("order_type",             "LIMIT")).upper()
        self.slip_pct          = float(cfg.get("limit_slippage_pct",   0.002))
        self.portfolio_value   = float(cfg.get("portfolio_value",      100_000))
        self.dry_run           = bool(cfg.get("dry_run",               True))

        self._ib: Optional["IB"] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        rl_portfolio:      dict,
        ranking_output:    dict,
        crash_result:      dict,
        featured_data:     Dict,
    ) -> dict:
        """
        Parameters
        ----------
        rl_portfolio   : output of RLTradingAgent.predict()
        ranking_output : output of RankingAgent.run()
        crash_result   : output of CrashDetectionAgent.run()
        featured_data  : {ticker → enriched DataFrame} for latest prices

        Returns
        -------
        {
            "executed_trades": [trade_dict, ...],
            "status":          "ok" | "kill_switch" | "disabled" | "dry_run",
            "message":         str,
        }
        """
        crash_prob = float(crash_result.get("crash_probability", 0.0))
        regime     = crash_result.get("regime", "neutral")

        # ── Master checks ─────────────────────────────────────────────────────
        if not self.enabled:
            logger.info("IBKR agent disabled in config")
            return self._no_op("disabled", "IBKR agent disabled in config")

        if crash_prob >= self.kill_threshold:
            logger.warning(
                "KILL SWITCH ACTIVATED — crash_prob=%.3f ≥ threshold=%.3f  "
                "regime=%s  No trades placed.",
                crash_prob, self.kill_threshold, regime,
            )
            return self._no_op(
                "kill_switch",
                f"Kill-switch active: crash_prob={crash_prob:.3f}, regime={regime}",
            )

        # ── Build target weights from RL + ranking signals ────────────────────
        target_weights = self._merge_signals(rl_portfolio, ranking_output, crash_prob)

        # ── Get latest prices ─────────────────────────────────────────────────
        latest_prices = self._get_prices(featured_data, target_weights.keys())

        # ── Compute orders ────────────────────────────────────────────────────
        orders = self._compute_orders(target_weights, latest_prices)

        if not orders:
            return self._no_op("ok", "No actionable orders computed")

        # ── Execute ───────────────────────────────────────────────────────────
        if self.dry_run or not _IB_AVAILABLE:
            trades = self._dry_run_orders(orders, latest_prices)
        else:
            trades = self._execute_orders(orders, latest_prices)

        logger.info("Execution complete: %d trade(s) processed", len(trades))
        return {
            "executed_trades": [t.to_dict() for t in trades],
            "status":          "dry_run" if (self.dry_run or not _IB_AVAILABLE) else "ok",
            "message":         f"Processed {len(trades)} order(s); regime={regime}",
        }

    # ── Signal merging ────────────────────────────────────────────────────────

    def _merge_signals(
        self,
        rl_portfolio:   dict,
        ranking_output: dict,
        crash_prob:     float,
    ) -> Dict[str, float]:
        """Blend RL weights with ranking signals; scale by crash risk."""
        # RL weights
        rl_weights = rl_portfolio.get("weights", {})
        cash_w     = float(rl_portfolio.get("cash_weight", 0.0))

        # Boost top-ranked tickers if RL weight is low
        top_10 = ranking_output.get("final_top_10_diversified",
                 ranking_output.get("1_month", []))

        blended: Dict[str, float] = {}
        for t, w in rl_weights.items():
            rank_bonus = 0.02 if t in top_10 else 0.0
            blended[t] = float(w) + rank_bonus

        # Reduce exposure proportional to crash risk
        equity_scale = max(0.0, 1.0 - crash_prob * 1.5)
        blended = {t: w * equity_scale for t, w in blended.items()}

        # Enforce per-position cap
        blended = {t: min(w, self.max_position_pct) for t, w in blended.items()}

        # Enforce total exposure cap
        total = sum(blended.values())
        if total > self.max_exposure:
            scale = self.max_exposure / total
            blended = {t: w * scale for t, w in blended.items()}

        return {t: round(w, 4) for t, w in blended.items() if w > 0.001}

    # ── Order computation ─────────────────────────────────────────────────────

    def _compute_orders(
        self,
        target_weights: Dict[str, float],
        prices:         Dict[str, float],
    ) -> List[dict]:
        """Convert target weights → share quantities."""
        orders = []
        for ticker, weight in target_weights.items():
            price = prices.get(ticker)
            if not price or price <= 0:
                logger.warning("No price available for %s — skipping", ticker)
                continue
            usd_alloc = self.portfolio_value * weight
            qty       = max(1, int(usd_alloc / price))
            orders.append({"ticker": ticker, "action": "BUY", "qty": qty, "price": price})
        return orders

    # ── Dry-run ───────────────────────────────────────────────────────────────

    def _dry_run_orders(
        self,
        orders: List[dict],
        prices: Dict[str, float],
    ) -> List[_TradeRecord]:
        trades = []
        for o in orders:
            lp = round(o["price"] * (1 + self.slip_pct), 2) if self.order_type == "LIMIT" else None
            rec = _TradeRecord(
                ticker=o["ticker"],
                action=o["action"],
                quantity=o["qty"],
                order_type=self.order_type,
                limit_price=lp,
                status="DRY_RUN",
                note="Dry-run — not submitted to broker",
            )
            trades.append(rec)
            logger.info(
                "DRY RUN | %s %d × %s @ %s",
                o["action"], o["qty"], o["ticker"],
                f"LIMIT {lp}" if lp else "MARKET",
            )
        return trades

    # ── Live execution (ib_insync) ─────────────────────────────────────────────

    def _execute_orders(
        self,
        orders: List[dict],
        prices: Dict[str, float],
    ) -> List[_TradeRecord]:
        if not self._ensure_connected():
            logger.error("Could not connect to IB Gateway — falling back to dry-run")
            return self._dry_run_orders(orders, prices)

        trades = []
        for o in orders:
            try:
                contract = Stock(o["ticker"], "SMART", "USD")
                self._ib.qualifyContracts(contract)

                if self.order_type == "LIMIT":
                    lp    = round(o["price"] * (1 + self.slip_pct), 2)
                    order = LimitOrder(o["action"], o["qty"], lp)
                else:
                    order = MarketOrder(o["action"], o["qty"])

                trade  = self._ib.placeOrder(contract, order)
                self._ib.sleep(1)  # brief wait for ack
                status = trade.orderStatus.status if trade else "UNKNOWN"

                rec = _TradeRecord(
                    ticker=o["ticker"],
                    action=o["action"],
                    quantity=o["qty"],
                    order_type=self.order_type,
                    limit_price=order.lmtPrice if self.order_type == "LIMIT" else None,
                    status=status,
                    note=f"live_trading={self.live_trading}",
                )
                trades.append(rec)
                logger.info(
                    "Order placed: %s %d × %s  status=%s",
                    o["action"], o["qty"], o["ticker"], status,
                )
            except Exception as exc:
                logger.error("Order failed for %s: %s", o["ticker"], exc)
                trades.append(_TradeRecord(
                    o["ticker"], o["action"], o["qty"],
                    status="REJECTED", note=str(exc),
                ))

        self._disconnect()
        return trades

    def _ensure_connected(self) -> bool:
        if not _IB_AVAILABLE:
            return False
        try:
            if self._ib is None:
                self._ib = IB()
            if not self._ib.isConnected():
                port = self.live_port if self.live_trading else self.paper_port
                self._ib.connect(
                    self.host, port, clientId=self.client_id, readonly=False
                )
                logger.info(
                    "Connected to IB %s port %d",
                    "LIVE" if self.live_trading else "PAPER", port,
                )
            return self._ib.isConnected()
        except Exception as exc:
            logger.error("IB connection failed: %s", exc)
            return False

    def _disconnect(self):
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

    # ── Price lookup ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_prices(
        featured_data: Dict,
        tickers,
    ) -> Dict[str, float]:
        prices = {}
        for t in tickers:
            df = featured_data.get(t)
            if df is not None and "Close" in df.columns:
                s = df["Close"].dropna()
                if not s.empty:
                    prices[t] = round(float(s.iloc[-1]), 4)
        return prices

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _no_op(status: str, message: str) -> dict:
        return {"executed_trades": [], "status": status, "message": message}
