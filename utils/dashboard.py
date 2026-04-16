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
                f"MaxDD: {bt.get('max_drawdown',0):.1%}  "
                f"Win Rate: {bt.get('win_rate',0):.1%}"
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
    html_parts.append('<nav class="nav-bar">')
    for i, (title, _) in enumerate(figs):
        active = " active" if i == 0 else ""
        num = f"0{i+1}" if i < 9 else str(i+1)
        html_parts.append(
            f'<button class="tab{active}" onclick="showTab({i})">'
            f'<span class="tab-num">{num}</span>{title}</button>'
        )
    html_parts.append("</nav>")

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
    palette = ["#00e5a0", "#00b87a", "#f0b429", "#e05c5c", "#9b72cf"]
    return [(palette[i] if i < len(lst) else "#0d1526") for i in range(n)]


def _html_header(now: str, meta: dict) -> str:
    models_str = ', '.join(meta.get('models_used', []))
    n_tickers  = meta.get('n_tickers_analysed', '?')
    elapsed    = meta.get('elapsed_seconds', '?')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Deep Market — Stock Forecast Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    /* ── Design tokens ─────────────────────────────────────── */
    :root {{
      --bg-void:      #04060e;
      --bg-deep:      #080c18;
      --bg-surface:   #0d1526;
      --bg-elevated:  #111c33;
      --border-dim:   #1a2744;
      --border-glow:  #00e5a044;
      --mint:         #00e5a0;
      --mint-dim:     #00b87a;
      --amber:        #f0b429;
      --crimson:      #ff4757;
      --violet:       #9b72cf;
      --text-primary: #e2e8f8;
      --text-secondary:#8899bb;
      --text-muted:   #4a5878;
      --font-display: 'Bebas Neue', sans-serif;
      --font-mono:    'IBM Plex Mono', monospace;
      --font-body:    'IBM Plex Sans', sans-serif;
    }}

    /* ── Reset & base ──────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    html {{ scroll-behavior: smooth; }}

    body {{
      background: var(--bg-void);
      color: var(--text-primary);
      font-family: var(--font-body);
      min-height: 100vh;
      overflow-x: hidden;
      position: relative;
    }}

    /* ── Scanline texture overlay ──────────────────────────── */
    body::before {{
      content: '';
      position: fixed; inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
      );
      pointer-events: none;
      z-index: 9999;
    }}

    /* ── Radial ambient glow ───────────────────────────────── */
    body::after {{
      content: '';
      position: fixed;
      top: -20vh; left: 50%;
      transform: translateX(-50%);
      width: 80vw; height: 60vh;
      background: radial-gradient(ellipse at center,
        rgba(0,229,160,0.04) 0%,
        rgba(0,229,160,0.01) 50%,
        transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}

    /* ── Ticker tape strip ─────────────────────────────────── */
    .ticker-tape {{
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-dim);
      padding: 6px 0;
      overflow: hidden;
      white-space: nowrap;
    }}
    .ticker-inner {{
      display: inline-block;
      animation: ticker-scroll 32s linear infinite;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-secondary);
      letter-spacing: 0.08em;
    }}
    .ticker-inner .up   {{ color: var(--mint); }}
    .ticker-inner .down {{ color: var(--crimson); }}
    .ticker-inner .sep  {{ margin: 0 24px; color: var(--border-dim); }}
    @keyframes ticker-scroll {{
      0%   {{ transform: translateX(0); }}
      100% {{ transform: translateX(-50%); }}
    }}

    /* ── Main layout wrapper ───────────────────────────────── */
    .shell {{
      position: relative;
      z-index: 1;
      max-width: 1600px;
      margin: 0 auto;
      padding: 0 24px 48px;
    }}

    /* ── Header ────────────────────────────────────────────── */
    .site-header {{
      padding: 40px 0 32px;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: end;
      gap: 24px;
      border-bottom: 1px solid var(--border-dim);
      margin-bottom: 32px;
      position: relative;
    }}
    .site-header::after {{
      content: '';
      position: absolute;
      bottom: -1px; left: 0;
      width: 180px; height: 1px;
      background: linear-gradient(90deg, var(--mint), transparent);
    }}

    .header-wordmark {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .wordmark-eyebrow {{
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: 0.3em;
      text-transform: uppercase;
      color: var(--mint);
    }}
    .wordmark-title {{
      font-family: var(--font-display);
      font-size: clamp(2.8rem, 5vw, 4.8rem);
      line-height: 0.9;
      letter-spacing: 0.04em;
      color: var(--text-primary);
      text-shadow: 0 0 60px rgba(0,229,160,0.15);
    }}
    .wordmark-sub {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      letter-spacing: 0.12em;
      margin-top: 8px;
    }}

    .header-stats {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 10px;
    }}
    .stat-row {{
      display: flex;
      gap: 20px;
    }}
    .stat-pill {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background: var(--bg-surface);
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      padding: 8px 16px;
      min-width: 80px;
    }}
    .stat-pill .val {{
      font-family: var(--font-display);
      font-size: 1.5rem;
      color: var(--mint);
      line-height: 1;
    }}
    .stat-pill .lbl {{
      font-family: var(--font-mono);
      font-size: 9px;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .header-timestamp {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      letter-spacing: 0.1em;
    }}
    .header-models {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-secondary);
      letter-spacing: 0.05em;
    }}
    .model-tag {{
      display: inline-block;
      background: var(--bg-elevated);
      border: 1px solid var(--border-dim);
      border-radius: 3px;
      padding: 2px 8px;
      margin-left: 6px;
      color: var(--amber);
    }}

    /* ── Tab navigation ────────────────────────────────────── */
    .nav-bar {{
      display: flex;
      gap: 2px;
      margin-bottom: 28px;
      border-bottom: 1px solid var(--border-dim);
      overflow-x: auto;
      scrollbar-width: none;
    }}
    .nav-bar::-webkit-scrollbar {{ display: none; }}

    .tab {{
      position: relative;
      background: transparent;
      color: var(--text-muted);
      border: none;
      padding: 12px 22px 14px;
      cursor: pointer;
      font-family: var(--font-mono);
      font-size: 11px;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      white-space: nowrap;
      transition: color 0.2s;
      outline: none;
    }}
    .tab::after {{
      content: '';
      position: absolute;
      bottom: -1px; left: 0; right: 0;
      height: 2px;
      background: var(--mint);
      transform: scaleX(0);
      transition: transform 0.25s cubic-bezier(0.4,0,0.2,1);
    }}
    .tab:hover {{ color: var(--text-secondary); }}
    .tab.active {{
      color: var(--mint);
    }}
    .tab.active::after {{
      transform: scaleX(1);
    }}

    /* Tab counter badge */
    .tab-num {{
      display: inline-block;
      font-size: 9px;
      color: var(--text-muted);
      margin-right: 7px;
      vertical-align: 1px;
    }}
    .tab.active .tab-num {{ color: var(--mint-dim); }}

    /* ── Content panels ────────────────────────────────────── */
    .tab-content {{
      width: 100%;
      animation: panel-in 0.3s cubic-bezier(0.4,0,0.2,1);
    }}
    @keyframes panel-in {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── Plotly chart container overrides ─────────────────── */
    .js-plotly-plot .plotly {{
      border-radius: 6px;
      overflow: hidden;
    }}

    /* ── Scrollbar styling ─────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-void); }}
    ::-webkit-scrollbar-thumb {{
      background: var(--border-dim);
      border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--mint-dim); }}

    /* ── Section label (above charts) ─────────────────────── */
    .section-label {{
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--text-muted);
      padding: 0 2px 12px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .section-label::after {{
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border-dim);
    }}

    /* ── Responsive ────────────────────────────────────────── */
    @media (max-width: 768px) {{
      .site-header {{ grid-template-columns: 1fr; }}
      .header-stats {{ align-items: flex-start; }}
      .stat-row {{ flex-wrap: wrap; }}
    }}
  </style>

  <script>
    function showTab(idx) {{
      document.querySelectorAll('.tab-content').forEach((el, i) => {{
        el.style.display = (i === idx) ? 'block' : 'none';
        if (i === idx) {{
          el.style.animation = 'none';
          el.offsetHeight;  /* reflow */
          el.style.animation = '';
        }}
      }});
      document.querySelectorAll('.tab').forEach((btn, i) => {{
        btn.classList.toggle('active', i === idx);
      }});
    }}
  </script>
</head>
<body>

<!-- Ticker tape -->
<div class="ticker-tape">
  <div class="ticker-inner">
    <span class="up">▲ TimesFM-2.5&nbsp;&nbsp;GOOGLE AI</span>
    <span class="sep">|</span>
    <span class="up">▲ Chronos-Bolt&nbsp;&nbsp;AMAZON AI</span>
    <span class="sep">|</span>
    <span>FinBERT&nbsp;&nbsp;SENTIMENT ANALYSIS</span>
    <span class="sep">|</span>
    <span>MACRO REGIME&nbsp;&nbsp;<span class="up">NEUTRAL {meta.get('macro_score_label','0.55')}</span></span>
    <span class="sep">|</span>
    <span class="up">▲ FUNDAMENTALS&nbsp;&nbsp;P/E · EPS · ROE · D/E</span>
    <span class="sep">|</span>
    <span>TECHNICAL&nbsp;&nbsp;RSI · MACD · BOLLINGER</span>
    <span class="sep">|</span>
    <span>GENERATED&nbsp;&nbsp;{now}</span>
    <span class="sep">|</span>
    <!-- duplicate for seamless loop -->
    <span class="up">▲ TimesFM-2.5&nbsp;&nbsp;GOOGLE AI</span>
    <span class="sep">|</span>
    <span class="up">▲ Chronos-Bolt&nbsp;&nbsp;AMAZON AI</span>
    <span class="sep">|</span>
    <span>FinBERT&nbsp;&nbsp;SENTIMENT ANALYSIS</span>
    <span class="sep">|</span>
    <span>MACRO REGIME&nbsp;&nbsp;<span class="up">NEUTRAL</span></span>
    <span class="sep">|</span>
    <span class="up">▲ FUNDAMENTALS&nbsp;&nbsp;P/E · EPS · ROE · D/E</span>
    <span class="sep">|</span>
    <span>TECHNICAL&nbsp;&nbsp;RSI · MACD · BOLLINGER</span>
    <span class="sep">|</span>
    <span>GENERATED&nbsp;&nbsp;{now}</span>
    <span class="sep">|</span>
  </div>
</div>

<div class="shell">
  <!-- Header -->
  <header class="site-header">
    <div class="header-wordmark">
      <span class="wordmark-eyebrow">Multi-Agent AI Forecasting System</span>
      <h1 class="wordmark-title">DEEP MARKET</h1>
      <span class="wordmark-sub">INTELLIGENCE · PRECISION · SIGNAL</span>
    </div>
    <div class="header-stats">
      <div class="stat-row">
        <div class="stat-pill">
          <span class="val">{n_tickers}</span>
          <span class="lbl">Tickers</span>
        </div>
        <div class="stat-pill">
          <span class="val">{elapsed}s</span>
          <span class="lbl">Runtime</span>
        </div>
        <div class="stat-pill">
          <span class="val">6</span>
          <span class="lbl">Signals</span>
        </div>
      </div>
      <div class="header-models">
        Models:
        {''.join(f'<span class="model-tag">{m}</span>' for m in meta.get('models_used', []))}
      </div>
      <div class="header-timestamp">&#x25CF;&nbsp; {now}</div>
    </div>
  </header>
"""


def _html_footer() -> str:
    return """
</div><!-- /.shell -->
</body>
</html>"""
