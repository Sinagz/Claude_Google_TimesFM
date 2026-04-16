"""
Interactive HTML Dashboard  (H)
────────────────────────────────
Generates a self-contained outputs/dashboard.html using Plotly.

Sections
  1. Rankings — top-5 per horizon colour-coded table
  2. Price + Forecast — historical prices + TimesFM & Chronos lines per top ticker
  3. Score Heatmap — all tickers × horizons
  4. Signal Breakdown — stacked bars (6 signals) for top tickers
  5. Backtest Equity Curve — strategy vs benchmark
  6. Sentiment — horizontal bar chart all tickers
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("dashboard")


def generate_dashboard(
    results: Dict[str, Any],
    config: Dict[str, Any],
    raw_data: Dict[str, pd.DataFrame],
) -> str:
    """Build the dashboard HTML and write to outputs/dashboard.html.
    Returns the file path."""
    try:
        import plotly.graph_objects as go
        import plotly.figure_factory as ff
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        logger.error("plotly not installed. Run: pip install plotly")
        return ""

    out_dir = os.path.dirname(config["output_path"]) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dashboard.html")

    details  = results.get("details", {})
    m1       = results.get("1_month",  [])
    m6       = results.get("6_month",  [])
    m12      = results.get("1_year",   [])
    all_tops = list(dict.fromkeys(m1 + m6 + m12))
    bt       = results.get("backtest", {})
    meta     = results.get("run_metadata", {})

    figs = []

    # ── 1. Rankings Table ────────────────────────────────────────────────────
    max_n = max(len(m1), len(m6), len(m12), 1)
    fig_rank = go.Figure(data=[go.Table(
        header=dict(
            values=["Rank", "1 Month", "6 Months", "1 Year"],
            fill_color="#1a1a2e",
            font=dict(color="white", size=14, family="monospace"),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[
                [f"#{i+1}" for i in range(max_n)],
                [(m1[i]  if i < len(m1)  else "—") for i in range(max_n)],
                [(m6[i]  if i < len(m6)  else "—") for i in range(max_n)],
                [(m12[i] if i < len(m12) else "—") for i in range(max_n)],
            ],
            fill_color=[
                ["#16213e"] * max_n,
                _rank_colors(m1,  max_n),
                _rank_colors(m6,  max_n),
                _rank_colors(m12, max_n),
            ],
            font=dict(color="white", size=13, family="monospace"),
            align="center",
            height=30,
        ),
    )])
    fig_rank.update_layout(
        title="Top 5 Stock Rankings by Horizon",
        paper_bgcolor="#0f0f23", plot_bgcolor="#0f0f23",
        font_color="white", margin=dict(t=50, b=10),
        height=230,
    )
    figs.append(("Rankings", fig_rank))

    # ── 2. Price History + Forecast Lines ────────────────────────────────────
    for ticker in all_tops:
        df  = raw_data.get(ticker)
        d   = details.get(ticker, {})
        if df is None or df.empty or not d:
            continue

        # Last 180 days of history
        hist = df["Close"].dropna().tail(180)
        if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        last_date  = pd.Timestamp(hist.index[-1])
        last_price = float(hist.iloc[-1])

        horizons = {"1 Month": (21, "1month"), "6 Months": (126, "6month"), "1 Year": (252, "1year")}

        fig_fc = go.Figure()

        # History line
        fig_fc.add_trace(go.Scatter(
            x=hist.index.tolist(), y=hist.tolist(),
            mode="lines", name="Price", line=dict(color="#00d4ff", width=2),
        ))

        for h_label, (h_days, h_key) in horizons.items():
            target_date = last_date + timedelta(days=int(h_days * 365 / 252))

            tfm = d.get(f"timesfm_{h_key}", {}) or {}
            chr_ = d.get(f"chronos_{h_key}", {}) or {}

            tfm_pt  = tfm.get("point")
            chr_pt  = chr_.get("point")
            chr_lo  = chr_.get("low")
            chr_hi  = chr_.get("high")

            if tfm_pt:
                fig_fc.add_trace(go.Scatter(
                    x=[last_date, target_date], y=[last_price, tfm_pt],
                    mode="lines+markers",
                    name=f"TimesFM {h_label}",
                    line=dict(dash="dot", width=2),
                    marker=dict(size=8),
                ))

            if chr_pt:
                fig_fc.add_trace(go.Scatter(
                    x=[last_date, target_date], y=[last_price, chr_pt],
                    mode="lines+markers",
                    name=f"Chronos {h_label}",
                    line=dict(dash="dash", width=2),
                    marker=dict(symbol="diamond", size=8),
                ))

            if chr_lo and chr_hi:
                fig_fc.add_trace(go.Scatter(
                    x=[target_date, target_date], y=[chr_lo, chr_hi],
                    mode="lines",
                    name=f"Chronos CI {h_label}",
                    line=dict(color="rgba(255,165,0,0.4)", width=6),
                    showlegend=False,
                ))

        fig_fc.update_layout(
            title=f"{ticker} — Price History & Forecasts",
            xaxis_title="Date", yaxis_title="Price (USD/CAD)",
            paper_bgcolor="#0f0f23", plot_bgcolor="#141428",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444"),
            hovermode="x unified",
            height=450,
        )
        figs.append((f"Forecast: {ticker}", fig_fc))

    # ── 3. Score Heatmap (all tickers × horizons) ────────────────────────────
    all_tickers_sorted = sorted(
        details.keys(),
        key=lambda t: details[t].get("score", 0),
        reverse=True,
    )
    horizons_labels = ["1 Month", "6 Months", "1 Year"]
    horizon_keys    = ["score_1month", "score_6month", "score_1year"]

    z    = [[details[t].get(hk, 0.5) for hk in horizon_keys] for t in all_tickers_sorted]
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z, x=horizons_labels, y=all_tickers_sorted,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar=dict(title="Score", tickfont=dict(color="white")),
    ))
    fig_heat.update_layout(
        title="Composite Score Heatmap — All Tickers × Horizons",
        paper_bgcolor="#0f0f23", plot_bgcolor="#141428",
        font_color="white",
        xaxis=dict(tickfont=dict(color="white")),
        yaxis=dict(tickfont=dict(color="white", size=10)),
        height=max(400, len(all_tickers_sorted) * 18 + 80),
        margin=dict(l=100),
    )
    figs.append(("Score Heatmap", fig_heat))

    # ── 4. Signal Breakdown (top tickers) ────────────────────────────────────
    signals_labels = ["TimesFM", "Chronos", "Technical", "Sentiment", "Fundamentals", "Macro"]
    signal_keys    = [
        "score_1month",         # proxy for timesfm signal (already fused)
        "agreement_score",
        "technical_score",
        "sentiment",
        "fundamentals_score",
        "macro_score",
    ]
    signal_colors = ["#00d4ff", "#ff6b6b", "#4ecdc4", "#f7dc6f", "#bb8fce", "#82e0aa"]

    fig_signals = go.Figure()
    for sig_label, sig_key, color in zip(signals_labels, signal_keys, signal_colors):
        values = []
        for t in all_tops:
            raw_val = details.get(t, {}).get(sig_key, 0.5)
            # Normalise sentiment from [-1,1] to [0,1]
            if sig_key == "sentiment":
                raw_val = (float(raw_val) + 1.0) / 2.0
            values.append(float(raw_val))

        fig_signals.add_trace(go.Bar(
            name=sig_label, x=all_tops, y=values,
            marker_color=color,
        ))

    fig_signals.update_layout(
        title="Signal Breakdown for Top-Ranked Tickers",
        barmode="group",
        xaxis_title="Ticker", yaxis_title="Score [0–1]",
        paper_bgcolor="#0f0f23", plot_bgcolor="#141428",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444"),
        yaxis=dict(range=[0, 1.05]),
        height=420,
    )
    figs.append(("Signal Breakdown", fig_signals))

    # ── 5. Backtest Equity Curve ──────────────────────────────────────────────
    if bt and bt.get("equity_curve"):
        ec   = bt["equity_curve"]
        dates  = [row[0] for row in ec]
        values = [row[1] for row in ec]
        bench  = float(bt.get("benchmark_return", 0))
        # Benchmark: straight line from 1.0 to 1+bench_return
        bench_vals = [1.0 + bench * i / max(len(dates) - 1, 1) for i in range(len(dates))]

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=dates, y=values,
            name="Strategy", mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ))
        fig_bt.add_trace(go.Scatter(
            x=dates, y=bench_vals,
            name="Buy-and-Hold Benchmark", mode="lines",
            line=dict(color="#f7dc6f", width=1.5, dash="dot"),
        ))
        fig_bt.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Start")
        fig_bt.update_layout(
            title=(
                f"Backtest Equity Curve  |  "
                f"Return: {bt.get('cumulative_return',0):.1%}  "
                f"Sharpe: {bt.get('sharpe_ratio',0):.2f}  "
                f"MaxDD: {bt.get('max_drawdown',0):.1%}"
            ),
            xaxis_title="Date", yaxis_title="Portfolio Value (normalised)",
            paper_bgcolor="#0f0f23", plot_bgcolor="#141428",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            hovermode="x unified",
            height=380,
        )
        figs.append(("Backtest", fig_bt))

    # ── 6. Sentiment Bar ──────────────────────────────────────────────────────
    sent_tickers = sorted(details.keys(), key=lambda t: details[t].get("sentiment", 0), reverse=True)
    sent_vals    = [float(details[t].get("sentiment", 0)) for t in sent_tickers]
    sent_colors  = ["#27ae60" if v >= 0 else "#e74c3c" for v in sent_vals]

    fig_sent = go.Figure(go.Bar(
        x=sent_vals, y=sent_tickers, orientation="h",
        marker_color=sent_colors,
        text=[f"{v:+.3f}" for v in sent_vals],
        textposition="auto",
    ))
    fig_sent.add_vline(x=0, line_color="white", line_dash="dash")
    fig_sent.update_layout(
        title="News Sentiment Score (FinBERT) — All Tickers",
        xaxis_title="Sentiment Score [-1 = bearish → +1 = bullish]",
        paper_bgcolor="#0f0f23", plot_bgcolor="#141428",
        font_color="white",
        height=max(400, len(sent_tickers) * 20 + 80),
        margin=dict(l=90),
    )
    figs.append(("Sentiment", fig_sent))

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_parts = [_html_header(now, meta)]

    # Navigation tabs
    html_parts.append('<div class="tabs">')
    for i, (title, _) in enumerate(figs):
        active = " active" if i == 0 else ""
        html_parts.append(
            f'<button class="tab{active}" onclick="showTab({i})">{title}</button>'
        )
    html_parts.append("</div>")

    # Tab content
    for i, (title, fig) in enumerate(figs):
        display = "block" if i == 0 else "none"
        div_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
        html_parts.append(
            f'<div class="tab-content" id="tab-{i}" style="display:{display}">'
            f"{div_html}</div>"
        )

    html_parts.append(_html_footer())

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    logger.info("Dashboard saved → %s", out_path)
    return out_path


# ── HTML scaffolding ──────────────────────────────────────────────────────────

def _rank_colors(lst, n):
    palette = ["#1a5276", "#154360", "#0e6655", "#4a235a", "#7b241c"]
    return [(palette[i] if i < len(lst) else "#1a1a2e") for i in range(n)]


def _html_header(now: str, meta: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Stock Forecast Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0f0f23;
      color: #e0e0e0;
      font-family: 'Segoe UI', monospace;
      padding: 20px;
    }}
    header {{
      text-align: center;
      padding: 20px 0 10px;
      border-bottom: 1px solid #333;
      margin-bottom: 16px;
    }}
    header h1 {{ font-size: 1.7em; color: #00d4ff; letter-spacing: 1px; }}
    header p  {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
    .tabs {{
      display: flex; flex-wrap: wrap; gap: 6px;
      margin-bottom: 14px;
    }}
    .tab {{
      background: #1a1a2e; color: #aaa;
      border: 1px solid #333; border-radius: 6px;
      padding: 7px 14px; cursor: pointer; font-size: 0.85em;
      transition: all 0.2s;
    }}
    .tab:hover, .tab.active {{
      background: #00d4ff; color: #000; border-color: #00d4ff; font-weight: 600;
    }}
    .tab-content {{ width: 100%; }}
  </style>
  <script>
    function showTab(idx) {{
      document.querySelectorAll('.tab-content').forEach((el, i) => {{
        el.style.display = (i === idx) ? 'block' : 'none';
      }});
      document.querySelectorAll('.tab').forEach((btn, i) => {{
        btn.classList.toggle('active', i === idx);
      }});
    }}
  </script>
</head>
<body>
<header>
  <h1>Multi-Agent Stock Forecasting Dashboard</h1>
  <p>Generated: {now} &nbsp;|&nbsp;
     Tickers: {meta.get('n_tickers_analysed','?')} &nbsp;|&nbsp;
     Models: {', '.join(meta.get('models_used', []))} &nbsp;|&nbsp;
     Runtime: {meta.get('elapsed_seconds','?')} s
  </p>
</header>
"""


def _html_footer() -> str:
    return "</body></html>"
