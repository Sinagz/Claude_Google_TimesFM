"""
Visual Report Generator
────────────────────────
Renders a rich, colour-coded terminal report from the pipeline results dict
and also saves a plain-text copy to outputs/report.txt.

Uses the `rich` library (installed as a transitive dep of transformers/typer).
"""

import os
from datetime import datetime
from typing import Any, Dict

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v, decimals=2):
    try:
        return f"{float(v):+.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"

def _price(v):
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "N/A"

def _score_bar(score: float, width: int = 12) -> str:
    """ASCII progress bar for a [0,1] score."""
    filled = max(0, min(width, round(float(score) * width)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {float(score):.2f}"

def _agreement_color(score: float) -> str:
    if score >= 0.85:
        return "green"
    elif score >= 0.5:
        return "yellow"
    return "red"

def _change_color(pct: float) -> str:
    return "green" if pct >= 0 else "red"

# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Print rich report to terminal and save plain copy to outputs/report.txt."""

    # Write to both stdout (colour) and a plain file simultaneously
    out_dir = os.path.dirname(config["output_path"]) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.txt")

    # force_terminal + legacy_windows=False keeps ANSI mode on Windows cmd/PowerShell
    # and avoids the win32 renderer which chokes on non-cp1252 characters.
    import sys, io
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = Console(highlight=False, force_terminal=True,
                      legacy_windows=False, file=stdout_utf8)
    file_console = Console(file=open(report_path, "w", encoding="utf-8"),
                           highlight=False, no_color=True, width=90)

    def _print(*args, **kwargs):
        console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)

    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta = results.get("run_metadata", {})

    # ── Header ────────────────────────────────────────────────────────────────
    _print()
    _print(Panel(
        f"[bold cyan]Multi-Agent Stock Forecasting System[/bold cyan]\n"
        f"[dim]Run date: {now}   |   Tickers analysed: "
        f"{meta.get('n_tickers_analysed', '?')}   |   "
        f"Elapsed: {meta.get('elapsed_seconds', '?')} s[/dim]\n"
        f"[dim]Models: {', '.join(meta.get('models_used', []))}[/dim]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        expand=False,
    ))

    # ── Top-5 Rankings table ──────────────────────────────────────────────────
    _print("\n[bold white]TOP 5 STOCKS BY HORIZON[/bold white]")
    rank_table = Table(box=box.ROUNDED, border_style="bright_blue", show_lines=True)
    rank_table.add_column("Rank", style="bold yellow", width=6, justify="center")
    rank_table.add_column("1 Month",  style="bold green",  width=14, justify="center")
    rank_table.add_column("6 Months", style="bold cyan",   width=14, justify="center")
    rank_table.add_column("1 Year",   style="bold magenta",width=14, justify="center")

    m1  = results.get("1_month", [])
    m6  = results.get("6_month", [])
    m12 = results.get("1_year",  [])

    medals = [" #1 ", " #2 ", " #3 ", " #4 ", " #5 "]
    for i in range(5):
        rank_table.add_row(
            medals[i],
            m1[i]  if i < len(m1)  else "—",
            m6[i]  if i < len(m6)  else "—",
            m12[i] if i < len(m12) else "—",
        )
    _print(rank_table)

    # ── Detail cards for every ticker in any top-5 list ───────────────────────
    interesting = dict.fromkeys(m1 + m6 + m12)   # preserves order, deduplicates
    details = results.get("details", {})

    _print("\n[bold white]STOCK DETAILS[/bold white]")

    for ticker in interesting:
        d = details.get(ticker, {})
        if not d:
            continue

        # Determine which lists this ticker appears in
        tags = []
        if ticker in m1:  tags.append("[green]1M[/green]")
        if ticker in m6:  tags.append("[cyan]6M[/cyan]")
        if ticker in m12: tags.append("[magenta]1Y[/magenta]")
        tag_str = "  ".join(tags)

        detail_table = Table(box=box.SIMPLE, show_header=False,
                             padding=(0, 1), expand=False)
        detail_table.add_column("Key",   style="dim",   width=26)
        detail_table.add_column("Value", style="white", width=55)

        # Latest price
        lp = d.get("latest_price")
        detail_table.add_row("Latest price", _price(lp) if lp else "N/A")
        detail_table.add_row("Latest date",  str(d.get("latest_date", "N/A")))
        detail_table.add_row("", "")

        # Composite scores
        detail_table.add_row("Overall score",    _score_bar(d.get("score", 0.5)))
        detail_table.add_row("  1-month score",  _score_bar(d.get("score_1month", 0.5)))
        detail_table.add_row("  6-month score",  _score_bar(d.get("score_6month", 0.5)))
        detail_table.add_row("  1-year score",   _score_bar(d.get("score_1year",  0.5)))
        detail_table.add_row("Technical score",  _score_bar(d.get("technical_score", 0.5)))
        detail_table.add_row("Sentiment",        f"{float(d.get('sentiment', 0)):+.3f}  "
                                                  f"({'bullish' if float(d.get('sentiment',0))>=0 else 'bearish'})")
        detail_table.add_row("", "")

        # Model agreement
        agr  = float(d.get("agreement_score", 0.5))
        warn = d.get("divergence_warning", False)
        agr_color = _agreement_color(agr)
        detail_table.add_row(
            "Model agreement",
            f"[{agr_color}]{agr:.0%}[/{agr_color}]"
            + ("  [bold red][DIVERGENCE WARNING][/bold red]" if warn else ""),
        )
        detail_table.add_row("", "")

        # Forecast table
        fc_table = Table(box=box.MINIMAL_HEAVY_HEAD, show_lines=False,
                         padding=(0, 1), expand=False)
        fc_table.add_column("Horizon",          style="bold",  width=10)
        fc_table.add_column("TimesFM point",    width=14, justify="right")
        fc_table.add_column("TimesFM %",        width=10, justify="right")
        fc_table.add_column("Chronos point",    width=14, justify="right")
        fc_table.add_column("Chronos %",        width=10, justify="right")
        fc_table.add_column("Low / High",       width=22, justify="center")

        horizon_map = [("1 Month", "1month"), ("6 Months", "6month"), ("1 Year", "1year")]
        for h_label, h_key in horizon_map:
            tfm = d.get(f"timesfm_{h_key}", {}) or {}
            chr_ = d.get(f"chronos_{h_key}", {}) or {}

            tfm_pct  = float(tfm.get("pct_change", 0) or 0)
            chr_pct  = float(chr_.get("pct_change", 0) or 0)
            tfm_col  = _change_color(tfm_pct)
            chr_col  = _change_color(chr_pct)

            lo  = chr_.get("low")
            hi  = chr_.get("high")
            rng = f"{_price(lo)} – {_price(hi)}" if lo and hi else "N/A"

            fc_table.add_row(
                h_label,
                _price(tfm.get("point")),
                f"[{tfm_col}]{_pct(tfm_pct)}[/{tfm_col}]",
                _price(chr_.get("point")),
                f"[{chr_col}]{_pct(chr_pct)}[/{chr_col}]",
                rng,
            )

        _print(Panel(
            detail_table,
            title=f"[bold yellow]{ticker}[/bold yellow]  {tag_str}",
            border_style="yellow",
            expand=False,
        ))
        _print(fc_table)
        _print()

    # ── Backtest ──────────────────────────────────────────────────────────────
    bt = results.get("backtest", {})
    if bt and bt.get("cumulative_return") is not None:
        _print("\n[bold white]BACKTEST RESULTS[/bold white]"
               f"[dim]  (last {config['backtest']['lookback_days']} calendar days)[/dim]")

        bt_table = Table(box=box.ROUNDED, border_style="bright_blue",
                         show_lines=False, expand=False)
        bt_table.add_column("Metric",   style="dim",   width=30)
        bt_table.add_column("Strategy", style="white", width=16, justify="right")
        bt_table.add_column("Benchmark",style="white", width=16, justify="right")

        cr  = float(bt.get("cumulative_return", 0))
        br  = float(bt.get("benchmark_return",  0))
        ann = float(bt.get("annualised_return",  0))
        shr = float(bt.get("sharpe_ratio",       0))
        mdd = float(bt.get("max_drawdown",       0))
        alp = float(bt.get("strategy_vs_bench",  0))
        wr  = float(bt.get("win_rate",           0))

        cr_col  = _change_color(cr)
        alp_col = _change_color(alp)

        bt_table.add_row("Cumulative return",
                         f"[{cr_col}]{cr:+.2%}[/{cr_col}]",
                         f"{br:+.2%}")
        bt_table.add_row("Annualised return",
                         f"{ann:+.2%}", "—")
        bt_table.add_row("Alpha (vs. benchmark)",
                         f"[{alp_col}]{alp:+.2%}[/{alp_col}]", "—")
        bt_table.add_row("Sharpe ratio",
                         f"{'[green]' if shr>1 else '[yellow]' if shr>0 else '[red]'}"
                         f"{shr:.2f}"
                         f"{'[/green]' if shr>1 else '[/yellow]' if shr>0 else '[/red]'}",
                         "—")
        bt_table.add_row("Max drawdown",
                         f"[red]{mdd:.2%}[/red]", "—")
        bt_table.add_row("Win rate (daily)",
                         f"{'[green]' if wr>=0.55 else '[yellow]' if wr>=0.5 else '[red]'}"
                         f"{wr:.1%}"
                         f"{'[/green]' if wr>=0.55 else '[/yellow]' if wr>=0.5 else '[/red]'}",
                         "—")
        bt_table.add_row("Rebalances",
                         str(bt.get("n_rebalances", "—")), "—")

        _print(bt_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    _print(Panel(
        f"[dim]Full JSON  ->  {config['output_path']}\n"
        f"Text report ->  {report_path}[/dim]",
        border_style="dim",
        expand=False,
    ))
    _print()

    file_console.file.close()
    console.print(f"[dim]Report also saved to {report_path}[/dim]\n")
