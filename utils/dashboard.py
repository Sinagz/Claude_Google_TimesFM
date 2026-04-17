"""
Interactive HTML Dashboard
────────────────────────────
Generates a self-contained outputs/dashboard.html using Plotly.

Can be called two ways:
  1. From main pipeline:   generate_dashboard(results, config, raw_data)
  2. Standalone from JSON: python utils/dashboard.py
                           python utils/dashboard.py --json outputs/results.json

Tabs
  01  Overview          — regime card, crash gauge, recommendation, top-10
  02  Strategy Portfolios — growth / value / defensive tables
  03  RL Portfolio       — PPO allocation pie + bar
  04  Crash & Regime     — crash probability gauges + 5-signal breakdown
  05  Forecast: {ticker} — price history + TimesFM + Chronos per top ticker
  06  Score Heatmap      — all tickers × horizons
  07  Signal Breakdown   — 6-signal stacked bars for top tickers
  08  Risk Metrics       — alpha score, Sharpe proxy, drawdown risk, mispricing
  09  Backtest           — equity curve + Sortino/CAGR + regime performance
  10  Sentiment          — FinBERT sentiment all tickers
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard(
    results:  Dict[str, Any],
    config:   Dict[str, Any],
    raw_data: Dict[str, pd.DataFrame],
) -> str:
    """Build dashboard from in-memory results. Returns the HTML file path."""
    out_dir  = os.path.dirname(config.get("output_path", "outputs/results.json")) or "outputs"
    out_path = os.path.join(out_dir, "dashboard.html")
    _build(results, raw_data, out_path)
    return out_path


def generate_dashboard_from_json(
    json_path: str = "outputs/results.json",
    out_path:  Optional[str] = None,
) -> str:
    """Build dashboard from a saved results.json. Returns the HTML file path."""
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(json_path), "dashboard.html")
    _build(results, {}, out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def _build(
    results:  Dict[str, Any],
    raw_data: Dict[str, pd.DataFrame],
    out_path: str,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly not installed — run: pip install plotly")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    details  = results.get("details", {})
    m1       = results.get("1_month",  [])
    m6       = results.get("6_month",  [])
    m12      = results.get("1_year",   [])
    growth   = results.get("growth",   [])
    value    = results.get("value",    [])
    defend   = results.get("defensive",[])
    top10    = results.get("top_10_stocks", results.get("final_top_10_diversified", []))
    bt       = results.get("backtest", {})
    meta     = results.get("run_metadata", {})
    metrics  = results.get("metrics",  {})
    rl_port  = results.get("rl_portfolio", {})
    crash_p  = float(results.get("crash_risk", 0.0))
    regime   = results.get("market_regime", "neutral")
    rec      = results.get("recommendation", "HOLD")
    exp_ret  = results.get("expected_return", "N/A")

    all_tops     = list(dict.fromkeys(m1 + m6 + m12 + top10))
    all_strategy = list(dict.fromkeys(top10 + growth + value + defend))

    figs = []

    # ── Tab 01: Overview ─────────────────────────────────────────────────────
    regime_color = {"bull": "#00e5a0", "neutral": "#f0b429",
                    "high_volatility": "#ff9f43", "crash": "#ff4757"}.get(regime, "#8899bb")

    # Crash probability gauge
    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=round(crash_p * 100, 1),
        title={"text": "Crash Probability %", "font": {"size": 18, "color": "#e2e8f8", "family": "IBM Plex Mono"}},
        delta={"reference": 25, "suffix": "%"},
        number={"suffix": "%", "font": {"size": 42, "color": regime_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8899bb", "tickfont": {"color": "#8899bb"}},
            "bar":  {"color": regime_color, "thickness": 0.25},
            "bgcolor": "#0d1526",
            "bordercolor": "#1a2744",
            "steps": [
                {"range": [0,  25], "color": "rgba(0,229,160,0.12)"},
                {"range": [25, 50], "color": "rgba(240,180,41,0.12)"},
                {"range": [50, 75], "color": "rgba(255,159,67,0.12)"},
                {"range": [75,100], "color": "rgba(255,71,87,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#ff4757", "width": 3},
                "thickness": 0.75,
                "value": 65,
            },
        },
        domain={"x": [0, 0.45], "y": [0, 1]},
    ))

    # Recommendation indicator
    rec_color = {"BUY": "#00e5a0", "HOLD": "#f0b429", "REDUCE": "#ff9f43", "STOP": "#ff4757"}.get(rec, "#8899bb")
    fig_gauge.add_trace(go.Indicator(
        mode="number",
        value=None,
        title={"text": f"Recommendation<br><b style='font-size:28px;color:{rec_color}'>{rec}</b><br>"
                       f"<span style='font-size:13px;color:{regime_color}'>Regime: {regime.upper().replace('_',' ')}</span><br>"
                       f"<span style='font-size:13px;color:#8899bb'>Expected Return: {exp_ret}</span>",
               "font": {"size": 14, "color": "#e2e8f8", "family": "IBM Plex Mono"}},
        domain={"x": [0.55, 1], "y": [0.1, 0.9]},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#080c18", font_color="#e2e8f8",
        margin=dict(t=30, b=20, l=20, r=20), height=280,
    )
    figs.append(("Overview", fig_gauge))

    # Top-10 table
    max_n = max(len(top10), 1)
    sectors = [details.get(t, {}).get("sector", "—") for t in top10]
    alpha_s  = [f"{details.get(t,{}).get('alpha_score',0):+.4f}"  for t in top10]
    exp_rets = [f"{details.get(t,{}).get('expected_return',0):+.2%}" for t in top10]
    latest_p = [f"${details.get(t,{}).get('latest_price','—')}"   for t in top10]

    fig_top10 = go.Figure(data=[go.Table(
        columnwidth=[60, 140, 160, 120, 120, 120],
        header=dict(
            values=["#", "Ticker", "Sector", "Expected Ret", "Alpha Score", "Latest Price"],
            fill_color="#0d1526", font=dict(color="#00e5a0", size=13, family="IBM Plex Mono"),
            align="center", height=36,
        ),
        cells=dict(
            values=[
                [f"#{i+1}" for i in range(max_n)],
                top10[:max_n],
                sectors[:max_n],
                exp_rets[:max_n],
                alpha_s[:max_n],
                latest_p[:max_n],
            ],
            fill_color=["#080c18", "#0d1526"] * (max_n // 2 + 1),
            font=dict(color=["#8899bb", "#e2e8f8", "#8899bb", "#00e5a0", "#f0b429", "#e2e8f8"],
                      size=12, family="IBM Plex Mono"),
            align="center", height=28,
        ),
    )])
    fig_top10.update_layout(
        title="Final Top-10 Diversified Portfolio",
        paper_bgcolor="#080c18", font_color="#e2e8f8",
        margin=dict(t=50, b=10), height=max(300, max_n * 30 + 80),
    )
    figs.append(("Top 10", fig_top10))

    # ── Tab 02: Strategy Portfolios ──────────────────────────────────────────
    max_strat = max(len(growth), len(value), len(defend), 1)
    def _strat_col(lst, n):
        return [lst[i] if i < len(lst) else "—" for i in range(n)]

    fig_strat = go.Figure(data=[go.Table(
        columnwidth=[60, 140, 140, 140],
        header=dict(
            values=["#", "🚀 Growth", "💎 Value", "🛡 Defensive"],
            fill_color="#0d1526", font=dict(color=["#8899bb","#00e5a0","#00d4ff","#f0b429"],
                                            size=14, family="IBM Plex Mono"),
            align="center", height=40,
        ),
        cells=dict(
            values=[
                [f"#{i+1}" for i in range(max_strat)],
                _strat_col(growth,  max_strat),
                _strat_col(value,   max_strat),
                _strat_col(defend,  max_strat),
            ],
            fill_color=[["#080c18" if i%2==0 else "#0a0f1e" for i in range(max_strat)]]*4,
            font=dict(color=["#8899bb","#00e5a0","#00d4ff","#f0b429"],
                      size=13, family="IBM Plex Mono"),
            align="center", height=32,
        ),
    )])
    fig_strat.update_layout(
        title="Strategy Portfolios — Growth / Value / Defensive",
        paper_bgcolor="#080c18", font_color="#e2e8f8",
        margin=dict(t=50, b=10), height=max(320, max_strat * 34 + 90),
    )
    figs.append(("Strategies", fig_strat))

    # Sector distribution donut (from metrics)
    sect_dist = metrics.get("sector_distribution", {})
    if sect_dist:
        fig_sect = go.Figure(data=[go.Pie(
            labels=list(sect_dist.keys()),
            values=list(sect_dist.values()),
            hole=0.5,
            marker=dict(colors=_palette(len(sect_dist)),
                        line=dict(color="#080c18", width=2)),
            textfont=dict(family="IBM Plex Mono", size=11),
            textinfo="label+percent",
        )])
        fig_sect.update_layout(
            title="Final Top-10 Sector Distribution",
            paper_bgcolor="#080c18", font_color="#e2e8f8",
            showlegend=False, height=380,
            margin=dict(t=50, b=20),
        )
        figs.append(("Sector Mix", fig_sect))

    # ── Tab 03: RL Portfolio ─────────────────────────────────────────────────
    rl_weights = rl_port.get("weights", {})
    cash_w     = float(rl_port.get("cash_weight", 0.0))
    if rl_weights:
        all_rl = dict(sorted(rl_weights.items(), key=lambda x: -x[1]))
        if cash_w > 0:
            all_rl["CASH"] = cash_w

        labels = list(all_rl.keys())
        vals   = [round(v * 100, 2) for v in all_rl.values()]
        colors = ["#f0b429" if k == "CASH" else ("#00e5a0" if k in set(top10) else "#00d4ff")
                  for k in labels]

        fig_rl = go.Figure()
        fig_rl.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
            textfont=dict(family="IBM Plex Mono", size=11, color="#080c18"),
        ))
        fig_rl.update_layout(
            title=f"RL (PPO) Portfolio Allocation  |  Action: {rec}  |  Confidence: {float(rl_port.get('confidence',0)):.0%}",
            xaxis_title="Ticker", yaxis_title="Weight %",
            paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
            font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
            yaxis=dict(ticksuffix="%", gridcolor="#1a2744"),
            xaxis=dict(tickangle=-45),
            height=440,
        )
        figs.append(("RL Portfolio", fig_rl))

        # Pie version
        fig_rl_pie = go.Figure(data=[go.Pie(
            labels=labels, values=vals, hole=0.45,
            marker=dict(colors=colors, line=dict(color="#080c18", width=2)),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=11),
        )])
        fig_rl_pie.update_layout(
            title="RL Portfolio — Pie View",
            paper_bgcolor="#080c18", font_color="#e2e8f8",
            showlegend=False, height=420,
        )
        figs.append(("RL Pie", fig_rl_pie))

    # ── Tab 04: Crash & Regime ───────────────────────────────────────────────
    crash_signals = metrics.get("crash_signals", {})
    if crash_signals:
        sig_names  = list(crash_signals.keys())
        sig_vals   = [float(v) * 100 for v in crash_signals.values()]
        sig_colors = ["#ff4757" if v > 60 else "#ff9f43" if v > 35 else "#00e5a0"
                      for v in sig_vals]

        fig_crash = go.Figure()
        fig_crash.add_trace(go.Bar(
            x=sig_names, y=sig_vals,
            marker_color=sig_colors,
            text=[f"{v:.1f}%" for v in sig_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color="#e2e8f8"),
        ))
        fig_crash.add_hline(y=50, line_dash="dash", line_color="#ff4757",
                            annotation_text="Danger Zone (50%)",
                            annotation_font=dict(color="#ff4757", family="IBM Plex Mono"))
        fig_crash.add_hline(y=25, line_dash="dot", line_color="#f0b429",
                            annotation_text="Caution (25%)",
                            annotation_font=dict(color="#f0b429", family="IBM Plex Mono"))
        fig_crash.update_layout(
            title=f"Crash Detection Signals  |  Composite: {crash_p:.1%}  |  Regime: {regime.upper()}",
            xaxis_title="Signal", yaxis_title="Signal Intensity %",
            paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
            font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
            yaxis=dict(range=[0, 110], ticksuffix="%", gridcolor="#1a2744"),
            height=420,
        )
        figs.append(("Crash Risk", fig_crash))

    # ── Tab 05: Price + Forecast ─────────────────────────────────────────────
    forecast_tickers = list(dict.fromkeys(top10[:5] + m1[:3]))
    for ticker in forecast_tickers:
        df  = raw_data.get(ticker)
        d   = details.get(ticker, {})
        if not d:
            continue

        if df is not None and not df.empty and "Close" in df.columns:
            hist = df["Close"].dropna().tail(180)
            if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            x_hist  = hist.index.tolist()
            y_hist  = hist.tolist()
            last_dt = pd.Timestamp(hist.index[-1])
            last_px = float(hist.iloc[-1])
        else:
            x_hist, y_hist = [], []
            last_dt = pd.Timestamp.now()
            last_px = float(d.get("latest_price", 100))

        horizons = {"1 Month": (21, "1month"), "6 Months": (126, "6month"), "1 Year": (252, "1year")}
        tfm_colors  = ["#00d4ff", "#00a8cc", "#006b8f"]
        chr_colors  = ["#ff9f43", "#e67e22", "#d35400"]
        ml_color    = "#9b72cf"

        fig_fc = go.Figure()
        if x_hist:
            fig_fc.add_trace(go.Scatter(
                x=x_hist, y=y_hist, mode="lines", name="Price",
                line=dict(color="#e2e8f8", width=2),
            ))

        for idx, (h_label, (h_days, h_key)) in enumerate(horizons.items()):
            target_dt = last_dt + timedelta(days=int(h_days * 365 / 252))
            tfm = d.get(f"timesfm_{h_key}", {}) or {}
            chr = d.get(f"chronos_{h_key}", {}) or {}
            ml  = d.get(f"ml_{h_key}",      {}) or {}

            tfm_pt  = tfm.get("point")
            chr_pt  = chr.get("point")
            chr_lo  = chr.get("low")
            chr_hi  = chr.get("high")
            ml_ret  = ml.get("predicted_return", 0.0) or 0.0
            ml_pt   = last_px * (1 + ml_ret)

            if tfm_pt:
                fig_fc.add_trace(go.Scatter(
                    x=[last_dt, target_dt], y=[last_px, tfm_pt],
                    mode="lines+markers", name=f"TimesFM {h_label}",
                    line=dict(dash="dot", color=tfm_colors[idx], width=2),
                    marker=dict(size=8, color=tfm_colors[idx]),
                ))
            if chr_pt:
                fig_fc.add_trace(go.Scatter(
                    x=[last_dt, target_dt], y=[last_px, chr_pt],
                    mode="lines+markers", name=f"Chronos {h_label}",
                    line=dict(dash="dash", color=chr_colors[idx], width=2),
                    marker=dict(symbol="diamond", size=8, color=chr_colors[idx]),
                ))
            if chr_lo and chr_hi:
                fig_fc.add_trace(go.Scatter(
                    x=[target_dt, target_dt], y=[chr_lo, chr_hi],
                    mode="lines", name=f"CI {h_label}",
                    line=dict(color=chr_colors[idx], width=8),
                    opacity=0.3, showlegend=False,
                ))
            if abs(ml_ret) > 0.001:
                fig_fc.add_trace(go.Scatter(
                    x=[last_dt, target_dt], y=[last_px, ml_pt],
                    mode="lines+markers", name=f"ML {h_label}",
                    line=dict(dash="dashdot", color=ml_color, width=1.5),
                    marker=dict(symbol="cross", size=7, color=ml_color),
                    visible="legendonly",
                ))

        sector  = d.get("sector", "")
        alpha_v = d.get("alpha_score", 0.0)
        exp_r   = d.get("expected_return", 0.0)

        fig_fc.update_layout(
            title=f"{ticker}  [{sector}]  |  α={alpha_v:+.4f}  |  Expected: {exp_r:+.2%}  |  Latest: ${last_px:.2f}",
            xaxis_title="Date", yaxis_title="Price",
            paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
            font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
            legend=dict(bgcolor="rgba(8,12,24,0.8)", bordercolor="#1a2744",
                        font=dict(family="IBM Plex Mono", size=10)),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1a2744"),
            yaxis=dict(gridcolor="#1a2744"),
            height=480,
        )
        figs.append((f"📈 {ticker}", fig_fc))

    # ── Tab 06: Score Heatmap ────────────────────────────────────────────────
    sorted_tickers = sorted(details.keys(), key=lambda t: details[t].get("score", 0), reverse=True)
    h_labels = ["1 Month", "6 Months", "1 Year"]
    h_keys   = ["score_1month", "score_6month", "score_1year"]
    z    = [[details[t].get(k, 0.5) for k in h_keys] for t in sorted_tickers]
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=h_labels, y=sorted_tickers,
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#ff4757"],[0.5,"#f0b429"],[1,"#00e5a0"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Score", tickfont=dict(color="#8899bb", family="IBM Plex Mono")),
    ))
    fig_heat.update_layout(
        title="Composite Score Heatmap — All Tickers × Horizons",
        paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
        font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
        xaxis=dict(tickfont=dict(color="#8899bb")),
        yaxis=dict(tickfont=dict(color="#8899bb", size=9)),
        height=max(400, len(sorted_tickers) * 16 + 100),
        margin=dict(l=90),
    )
    figs.append(("Heatmap", fig_heat))

    # ── Tab 07: Signal Breakdown ─────────────────────────────────────────────
    sig_labels = ["Score 1M", "Agreement", "Technical", "Sentiment", "Fundamentals", "Macro"]
    sig_keys   = ["score_1month","agreement_score","technical_score","sentiment","fundamentals_score","macro_score"]
    sig_colors = ["#00e5a0","#00d4ff","#f0b429","#9b72cf","#ff9f43","#82e0aa"]

    fig_sig = go.Figure()
    for lbl, key, col in zip(sig_labels, sig_keys, sig_colors):
        vals = []
        for t in all_tops[:20]:
            v = float(details.get(t, {}).get(key, 0.5) or 0.5)
            if key == "sentiment":
                v = (v + 1.0) / 2.0
            vals.append(round(v, 3))
        fig_sig.add_trace(go.Bar(name=lbl, x=all_tops[:20], y=vals,
                                  marker_color=col))
    fig_sig.update_layout(
        title="Signal Breakdown — Top Tickers",
        barmode="group",
        xaxis_title="Ticker", yaxis_title="Score [0–1]",
        paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
        font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
        legend=dict(bgcolor="rgba(8,12,24,0.8)", bordercolor="#1a2744",
                    font=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(range=[0, 1.05], gridcolor="#1a2744"),
        xaxis=dict(tickangle=-35),
        height=440,
    )
    figs.append(("Signals", fig_sig))

    # ── Tab 08: Risk Metrics ─────────────────────────────────────────────────
    risk_tickers = [t for t in all_strategy if details.get(t)][:25]
    if risk_tickers:
        alpha_vals  = [float(details.get(t,{}).get("alpha_score",   0.0) or 0.0) for t in risk_tickers]
        sharpe_vals = [float(details.get(t,{}).get("sharpe_proxy",  0.0) or 0.0) for t in risk_tickers]
        misprice_v  = [float(details.get(t,{}).get("mispricing",    0.0) or 0.0) for t in risk_tickers]
        dd_vals     = [float(details.get(t,{}).get("drawdown_risk", 0.0) or 0.0) * 100 for t in risk_tickers]

        fig_risk = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Alpha Score", "Sharpe Proxy", "Mispricing", "Drawdown Risk %"],
            vertical_spacing=0.14, horizontal_spacing=0.10,
        )
        def _bar(vals, name, color, row, col):
            colors = [color if v >= 0 else "#ff4757" for v in vals]
            fig_risk.add_trace(go.Bar(
                x=risk_tickers, y=vals, name=name,
                marker_color=colors, showlegend=False,
            ), row=row, col=col)

        _bar(alpha_vals,  "Alpha",     "#00e5a0", 1, 1)
        _bar(sharpe_vals, "Sharpe",    "#00d4ff", 1, 2)
        _bar(misprice_v,  "Mispricing","#9b72cf", 2, 1)
        _bar([-v for v in dd_vals], "Drawdown", "#ff9f43", 2, 2)  # invert: lower = better

        fig_risk.update_layout(
            title="Risk Metrics — All Strategy Tickers",
            paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
            font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
            height=600,
        )
        for ann in fig_risk.layout.annotations:
            ann.font.color = "#8899bb"
            ann.font.family = "IBM Plex Mono"
        for axis in ["xaxis","xaxis2","xaxis3","xaxis4"]:
            fig_risk.layout[axis].update(tickangle=-40, tickfont=dict(size=9), gridcolor="#1a2744")
        for axis in ["yaxis","yaxis2","yaxis3","yaxis4"]:
            fig_risk.layout[axis].update(gridcolor="#1a2744")
        figs.append(("Risk Metrics", fig_risk))

    # ── Tab 09: Backtest ─────────────────────────────────────────────────────
    if bt and bt.get("equity_curve"):
        ec     = bt["equity_curve"]
        dates  = [r[0] for r in ec]
        vals   = [r[1] for r in ec]
        bench  = float(bt.get("benchmark_return", 0))
        bench_line = [1.0 + bench * i / max(len(dates) - 1, 1) for i in range(len(dates))]

        sharpe  = float(bt.get("sharpe_ratio",  0))
        sortino = float(bt.get("sortino_ratio", 0))
        cagr    = float(bt.get("cagr",          0))
        mdd     = float(bt.get("max_drawdown",  0))
        wr      = float(bt.get("win_rate",      0))
        cum_ret = float(bt.get("cumulative_return", 0))
        tc_bps  = float(bt.get("transaction_cost_bps", 10))
        slip_bps= float(bt.get("slippage_bps", 5))

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=dates, y=vals, name="Strategy",
            mode="lines", line=dict(color="#00e5a0", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,229,160,0.06)",
        ))
        fig_bt.add_trace(go.Scatter(
            x=dates, y=bench_line, name="Buy-and-Hold",
            mode="lines", line=dict(color="#f0b429", width=1.5, dash="dot"),
        ))
        fig_bt.add_hline(y=1.0, line_dash="dash", line_color="#4a5878",
                         annotation_text="Start", annotation_font=dict(color="#4a5878"))

        # Shade drawdown areas
        peak = vals[0]
        for i, v in enumerate(vals):
            peak = max(peak, v)

        fig_bt.update_layout(
            title=(
                f"Backtest Equity Curve  |  "
                f"CAGR: {cagr:+.1%}  Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}  "
                f"MaxDD: {mdd:.1%}  WinRate: {wr:.1%}  "
                f"TC: {tc_bps:.0f}bps  Slip: {slip_bps:.0f}bps"
            ),
            xaxis_title="Date", yaxis_title="Normalised Value",
            paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
            font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
            legend=dict(bgcolor="rgba(8,12,24,0.8)", bordercolor="#1a2744"),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1a2744"),
            yaxis=dict(gridcolor="#1a2744"),
            height=420,
        )
        figs.append(("Backtest", fig_bt))

        # Regime performance bar chart
        reg_perf = bt.get("regime_performance", {})
        if any(reg_perf.values()):
            r_names = [r for r, d in reg_perf.items() if d]
            r_rets  = [float(reg_perf[r].get("cum_return", 0)) * 100 for r in r_names]
            r_days  = [int(reg_perf[r].get("n_days", 0)) for r in r_names]
            r_cols  = ["#00e5a0" if v >= 0 else "#ff4757" for v in r_rets]

            fig_reg = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Cumulative Return % by Regime",
                                                    "Days Spent in Regime"])
            fig_reg.add_trace(go.Bar(x=r_names, y=r_rets, marker_color=r_cols,
                                     text=[f"{v:+.1f}%" for v in r_rets],
                                     textposition="auto", showlegend=False), row=1, col=1)
            fig_reg.add_trace(go.Bar(x=r_names, y=r_days, marker_color="#00d4ff",
                                     text=r_days, textposition="auto", showlegend=False), row=1, col=2)
            fig_reg.update_layout(
                paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
                font_color="#e2e8f8", font=dict(family="IBM Plex Mono"), height=340,
            )
            for ann in fig_reg.layout.annotations:
                ann.font.color = "#8899bb"; ann.font.family = "IBM Plex Mono"
            for axis in ["xaxis","xaxis2","yaxis","yaxis2"]:
                fig_reg.layout[axis].update(gridcolor="#1a2744")
            figs.append(("Regime Perf", fig_reg))

    # ── Tab 10: Sentiment ─────────────────────────────────────────────────────
    sent_tickers = sorted(details.keys(), key=lambda t: float(details[t].get("sentiment", 0) or 0), reverse=True)
    sent_vals    = [float(details[t].get("sentiment", 0) or 0) for t in sent_tickers]
    sent_colors  = ["#00e5a0" if v >= 0 else "#ff4757" for v in sent_vals]

    fig_sent = go.Figure(go.Bar(
        x=sent_vals, y=sent_tickers, orientation="h",
        marker_color=sent_colors,
        text=[f"{v:+.3f}" for v in sent_vals], textposition="auto",
        textfont=dict(family="IBM Plex Mono", size=10),
    ))
    fig_sent.add_vline(x=0, line_color="#4a5878", line_dash="dash")
    fig_sent.update_layout(
        title="FinBERT Sentiment Score  |  Bearish ← 0 → Bullish",
        xaxis_title="Sentiment [-1 → +1]",
        paper_bgcolor="#080c18", plot_bgcolor="#0d1526",
        font_color="#e2e8f8", font=dict(family="IBM Plex Mono"),
        xaxis=dict(gridcolor="#1a2744"),
        height=max(400, len(sent_tickers) * 18 + 100),
        margin=dict(l=90),
    )
    figs.append(("Sentiment", fig_sent))

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
    html_parts = [_html_header(now_str, meta, regime, crash_p, rec)]

    html_parts.append('<nav class="nav-bar">')
    for i, (title, _) in enumerate(figs):
        active = " active" if i == 0 else ""
        num = f"0{i+1}" if i < 9 else str(i+1)
        html_parts.append(
            f'<button class="tab{active}" onclick="showTab({i})">'
            f'<span class="tab-num">{num}</span>{title}</button>'
        )
    html_parts.append("</nav>")

    for i, (title, fig) in enumerate(figs):
        display  = "block" if i == 0 else "none"
        div_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
        html_parts.append(
            f'<div class="tab-content" id="tab-{i}" style="display:{display}">{div_html}</div>'
        )

    html_parts.append(_html_footer())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("Dashboard saved → %s  (%d tabs)", out_path, len(figs))


# ─────────────────────────────────────────────────────────────────────────────
# HTML scaffolding
# ─────────────────────────────────────────────────────────────────────────────

def _palette(n: int):
    base = ["#00e5a0","#00d4ff","#f0b429","#ff9f43","#9b72cf",
            "#ff4757","#82e0aa","#48c9b0","#f8c471","#bb8fce"]
    return [base[i % len(base)] for i in range(n)]


def _rank_colors(lst, n):
    pal = ["#00e5a0","#00b87a","#f0b429","#e05c5c","#9b72cf"]
    return [(pal[i] if i < len(lst) else "#0d1526") for i in range(n)]


def _html_header(now: str, meta: dict, regime: str, crash_p: float, rec: str) -> str:
    regime_color = {"bull":"#00e5a0","neutral":"#f0b429",
                    "high_volatility":"#ff9f43","crash":"#ff4757"}.get(regime,"#8899bb")
    rec_color = {"BUY":"#00e5a0","HOLD":"#f0b429","REDUCE":"#ff9f43","STOP":"#ff4757"}.get(rec,"#8899bb")
    models_str = "  ".join(f'<span class="model-tag">{m}</span>' for m in meta.get("models_used", []))
    n_tickers  = meta.get("n_tickers_analysed", "?")
    elapsed    = meta.get("elapsed_seconds",    "?")
    regime_label = regime.upper().replace("_"," ")

    # Build ticker-tape items from meta
    tape_items = [
        f'<span style="color:{regime_color}">▲ REGIME: {regime_label}</span>',
        f'<span class="sep">|</span>',
        f'<span style="color:{rec_color}">● RECOMMENDATION: {rec}</span>',
        f'<span class="sep">|</span>',
        f'<span style="color:{"#ff4757" if crash_p > 0.5 else "#00e5a0"}">CRASH PROB: {crash_p:.1%}</span>',
        '<span class="sep">|</span>',
        '<span>TimesFM-2.5 · Chronos-Bolt · XGBoost · PPO-RL · FinBERT</span>',
        '<span class="sep">|</span>',
        f'<span class="dim">GENERATED {now}</span>',
        '<span class="sep">|</span>',
        f'<span>TICKERS ANALYSED: {n_tickers}</span>',
        '<span class="sep">|</span>',
    ]
    tape_content = " ".join(tape_items) + " " + " ".join(tape_items)  # duplicate for loop

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Deep Market — Quant Trading Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-void:     #04060e;
      --bg-deep:     #080c18;
      --bg-surface:  #0d1526;
      --bg-elevated: #111c33;
      --border-dim:  #1a2744;
      --mint:        #00e5a0;
      --mint-dim:    #00b87a;
      --amber:       #f0b429;
      --crimson:     #ff4757;
      --violet:      #9b72cf;
      --text-1:      #e2e8f8;
      --text-2:      #8899bb;
      --text-3:      #4a5878;
      --mono:        'IBM Plex Mono', monospace;
      --sans:        'IBM Plex Sans', sans-serif;
      --display:     'Bebas Neue', sans-serif;
      --regime-color: {regime_color};
      --rec-color:    {rec_color};
    }}
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    html {{ scroll-behavior:smooth; }}
    body {{ background:var(--bg-void); color:var(--text-1); font-family:var(--sans); min-height:100vh; overflow-x:hidden; }}

    body::before {{
      content:''; position:fixed; inset:0;
      background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px);
      pointer-events:none; z-index:9999;
    }}
    body::after {{
      content:''; position:fixed; top:-20vh; left:50%; transform:translateX(-50%);
      width:80vw; height:60vh;
      background:radial-gradient(ellipse at center, rgba(0,229,160,0.04) 0%, transparent 70%);
      pointer-events:none; z-index:0;
    }}

    /* ── Ticker tape ── */
    .ticker-tape {{ background:var(--bg-surface); border-bottom:1px solid var(--border-dim); padding:7px 0; overflow:hidden; white-space:nowrap; }}
    .ticker-inner {{ display:inline-block; animation:scroll 45s linear infinite; font-family:var(--mono); font-size:11px; color:var(--text-2); letter-spacing:.06em; }}
    .sep {{ margin:0 20px; color:var(--border-dim); }}
    .dim {{ color:var(--text-3); }}
    @keyframes scroll {{ 0%{{transform:translateX(0)}} 100%{{transform:translateX(-50%)}} }}

    /* ── Layout ── */
    .shell {{ position:relative; z-index:1; max-width:1700px; margin:0 auto; padding:0 24px 60px; }}

    /* ── Header ── */
    .site-header {{
      padding:36px 0 28px; display:grid; grid-template-columns:1fr auto;
      align-items:end; gap:24px; border-bottom:1px solid var(--border-dim);
      margin-bottom:28px; position:relative;
    }}
    .site-header::after {{ content:''; position:absolute; bottom:-1px; left:0; width:200px; height:1px; background:linear-gradient(90deg,var(--mint),transparent); }}

    .wordmark {{ display:flex; flex-direction:column; gap:4px; }}
    .eyebrow {{ font-family:var(--mono); font-size:10px; letter-spacing:.3em; text-transform:uppercase; color:var(--mint); }}
    .title {{ font-family:var(--display); font-size:clamp(2.8rem,5vw,5rem); line-height:.88; letter-spacing:.04em; color:var(--text-1); text-shadow:0 0 80px rgba(0,229,160,0.14); }}
    .sub {{ font-family:var(--mono); font-size:11px; color:var(--text-3); letter-spacing:.12em; margin-top:6px; }}

    .header-right {{ display:flex; flex-direction:column; align-items:flex-end; gap:12px; }}
    .pills {{ display:flex; gap:12px; }}
    .pill {{ display:flex; flex-direction:column; align-items:center; background:var(--bg-surface); border:1px solid var(--border-dim); border-radius:4px; padding:8px 16px; min-width:90px; }}
    .pill .v {{ font-family:var(--display); font-size:1.6rem; line-height:1; }}
    .pill .l {{ font-family:var(--mono); font-size:9px; letter-spacing:.2em; text-transform:uppercase; color:var(--text-3); margin-top:4px; }}
    .regime-badge {{ display:inline-flex; align-items:center; gap:8px; background:var(--bg-elevated); border:1px solid var(--regime-color)44; border-radius:4px; padding:6px 14px; font-family:var(--mono); font-size:12px; color:var(--regime-color); letter-spacing:.1em; }}
    .rec-badge {{ background:var(--rec-color)22; border:1px solid var(--rec-color)88; border-radius:4px; padding:4px 12px; font-family:var(--display); font-size:1.4rem; color:var(--rec-color); letter-spacing:.08em; }}
    .ts {{ font-family:var(--mono); font-size:10px; color:var(--text-3); letter-spacing:.1em; }}
    .model-tag {{ display:inline-block; background:var(--bg-elevated); border:1px solid var(--border-dim); border-radius:3px; padding:2px 7px; margin-left:5px; color:var(--amber); font-family:var(--mono); font-size:10px; }}

    /* ── Nav bar ── */
    .nav-bar {{ display:flex; gap:2px; margin-bottom:24px; border-bottom:1px solid var(--border-dim); overflow-x:auto; scrollbar-width:none; }}
    .nav-bar::-webkit-scrollbar {{ display:none; }}
    .tab {{ position:relative; background:transparent; color:var(--text-3); border:none; padding:11px 20px 13px; cursor:pointer; font-family:var(--mono); font-size:11px; letter-spacing:.14em; text-transform:uppercase; white-space:nowrap; transition:color .2s; outline:none; }}
    .tab::after {{ content:''; position:absolute; bottom:-1px; left:0; right:0; height:2px; background:var(--mint); transform:scaleX(0); transition:transform .25s cubic-bezier(.4,0,.2,1); }}
    .tab:hover {{ color:var(--text-2); }}
    .tab.active {{ color:var(--mint); }}
    .tab.active::after {{ transform:scaleX(1); }}
    .tab-num {{ display:inline-block; font-size:9px; color:var(--text-3); margin-right:6px; vertical-align:1px; }}
    .tab.active .tab-num {{ color:var(--mint-dim); }}

    /* ── Content ── */
    .tab-content {{ width:100%; animation:in .28s cubic-bezier(.4,0,.2,1); }}
    @keyframes in {{ from{{opacity:0;transform:translateY(5px)}} to{{opacity:1;transform:translateY(0)}} }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track {{ background:var(--bg-void); }}
    ::-webkit-scrollbar-thumb {{ background:var(--border-dim); border-radius:3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background:var(--mint-dim); }}

    @media(max-width:768px) {{ .site-header{{grid-template-columns:1fr}} .header-right{{align-items:flex-start}} .pills{{flex-wrap:wrap}} }}
  </style>
  <script>
    function showTab(idx) {{
      document.querySelectorAll('.tab-content').forEach((el,i) => {{
        el.style.display = (i===idx)?'block':'none';
        if(i===idx){{ el.style.animation='none'; el.offsetHeight; el.style.animation=''; }}
      }});
      document.querySelectorAll('.tab').forEach((btn,i) => btn.classList.toggle('active',i===idx));
    }}
  </script>
</head>
<body>

<div class="ticker-tape">
  <div class="ticker-inner">{tape_content}</div>
</div>

<div class="shell">
  <header class="site-header">
    <div class="wordmark">
      <span class="eyebrow">Production Quant Trading Platform</span>
      <h1 class="title">DEEP MARKET</h1>
      <span class="sub">AI · SIGNAL · ALPHA · RISK · EXECUTION</span>
    </div>
    <div class="header-right">
      <div class="pills">
        <div class="pill"><span class="v" style="color:var(--mint)">{n_tickers}</span><span class="l">Tickers</span></div>
        <div class="pill"><span class="v" style="color:var(--amber)">{elapsed}s</span><span class="l">Runtime</span></div>
        <div class="pill"><span class="v" style="color:var(--regime-color)">{crash_p:.0%}</span><span class="l">Crash Risk</span></div>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <div class="regime-badge">◆ {regime_label}</div>
        <div class="rec-badge">{rec}</div>
      </div>
      <div style="font-family:var(--mono);font-size:10px;color:var(--text-2)">Models: {models_str}</div>
      <div class="ts">&#x25CF; {now}</div>
    </div>
  </header>
"""


def _html_footer() -> str:
    return "\n</div></body></html>"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate dashboard from results JSON")
    ap.add_argument("--json", default="outputs/results.json",
                    help="Path to results.json (default: outputs/results.json)")
    ap.add_argument("--out",  default=None,
                    help="Output HTML path (default: same dir as JSON)")
    args = ap.parse_args()

    print(f"Building dashboard from {args.json} ...")
    path = generate_dashboard_from_json(args.json, args.out)
    print(f"Dashboard saved -> {path}")
