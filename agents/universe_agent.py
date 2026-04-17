"""
Universe Agent
──────────────
Manages a large stock universe (~700+ tickers across all sectors) and applies
fast pre-screening to return a manageable, sector-diversified candidate list
for the full pipeline.

Pre-screening funnel
  1. Start from built-in universe (~700 tickers across 13 sectors)
  2. Bulk-download 20-day price/volume history via yfinance
  3. Filter: price > min_price, avg_volume > min_avg_volume
  4. Score each ticker: 0.5 × momentum_20d + 0.5 × volume_rank
  5. Take top K per sector → returns ≤ max_prescreen candidates

If the user has set a manual ticker override (config tickers < 30 tickers),
this agent returns None and the orchestrator uses the config list directly.

Exports
  get_ticker_sector(ticker) → str  — used by ranking and risk agents
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import get_all_tickers, setup_logger

logger = setup_logger("universe_agent")

# ── Full universe organized by sector ────────────────────────────────────────
# ~700 unique tickers; covers S&P 500, Russell 1000 additions, TSX, ETFs
_UNIVERSE: Dict[str, List[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "AMD", "ADBE", "CRM", "QCOM",
        "INTC", "MU", "TXN", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "ANSS",
        "FTNT", "PANW", "ZS", "CRWD", "NET", "DDOG", "SNOW", "PLTR", "TER",
        "MCHP", "MPWR", "ON", "SWKS", "ENPH", "FSLR", "CSCO", "IBM", "HPQ",
        "DELL", "HPE", "WDC", "STX", "NTAP", "PSTG", "ANET", "FFIV", "JNPR",
        "AKAM", "CDW", "CTSH", "EPAM", "GLOB", "CIEN", "COHR", "MKSI", "SLAB",
        "AMKR", "ICHR", "CRUS", "IDCC", "RMBS", "SYNA", "ACN", "INFY", "WIT",
    ],
    "Communication_Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "SNAP", "PINS", "RDDT", "MTCH", "EA", "TTWO", "RBLX", "U",
        "PARA", "WBD", "FOX", "NYT", "SPOT", "YELP", "ZM", "TWLO",
    ],
    "Healthcare": [
        "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "BMY", "AMGN", "GILD",
        "BIIB", "REGN", "VRTX", "MRNA", "CVS", "CI", "HUM", "MCK", "ABC",
        "DHR", "TMO", "ABT", "SYK", "BSX", "MDT", "EW", "ISRG", "ZBH", "BDX",
        "BAX", "HOLX", "IDXX", "IQV", "LH", "DGX", "VEEV", "NTRA", "GH",
        "ILMN", "NVAX", "SRPT", "ALNY", "BMRN", "INCY", "ACAD", "AXSM",
        "AGIO", "BPMC", "PKI", "WAT", "MTD", "HALO", "EXAS",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "COF",
        "USB", "TFC", "PNC", "KEY", "CFG", "FITB", "HBAN", "RF", "ZION",
        "MTB", "NTRS", "STT", "BK", "TROW", "IVZ", "V", "MA", "PYPL",
        "FIS", "FI", "GPN", "WU", "ICE", "CME", "CBOE", "NDAQ", "MKTX",
        "AFL", "MET", "PRU", "EQH", "AIG", "ALL", "PGR", "TRV", "CB",
        "MMC", "AON", "WTW", "AJG", "UPST", "LC", "SOFI", "NU", "AFRM",
    ],
    "Consumer_Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "ROST",
        "BURL", "ORLY", "AZO", "TSCO", "KMX", "AN", "PAG", "GPC", "LKQ",
        "LULU", "PVH", "VFC", "RL", "UAA", "URBN", "SKX", "DRI", "YUM",
        "QSR", "DPZ", "TXRH", "CMG", "MAR", "HLT", "H", "F", "GM",
        "BWA", "LEA", "ALV", "DHI", "LEN", "PHM", "NVR", "MDC", "LGIH",
        "ETSY", "W", "CHWY", "CZR", "MGM", "LVS", "WYNN", "PENN", "DKNG",
    ],
    "Consumer_Staples": [
        "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "KHC", "GIS", "K",
        "SJM", "CAG", "MKC", "HRL", "TSN", "HSY", "KR", "SFM", "BJ", "GO",
        "STZ", "SAM", "TAP", "MNST", "CELH", "EL", "CLX", "CHD", "KVUE",
    ],
    "Industrials": [
        "HON", "RTX", "LMT", "BA", "GE", "MMM", "CAT", "DE", "EMR", "ITW",
        "ETN", "PH", "ROK", "CMI", "XYL", "IR", "GWW", "MSC", "FAST", "NUE",
        "URI", "AMETEK", "GNRC", "SWK", "SNA", "TT", "JCI", "CARR", "OTIS",
        "FTV", "AXON", "ROP", "VRSK", "LDOS", "BAH", "SAIC", "LHX", "KTOS",
        "UPS", "FDX", "CHRW", "XPO", "SAIA", "ODFL", "JBHT", "KNX",
        "NSC", "CSX", "UNP", "DAL", "UAL", "AAL", "LUV", "ALK",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "MPC", "VLO", "PSX",
        "DVN", "PXD", "FANG", "OXY", "HES", "MRO", "APA", "CTRA", "SM",
        "KMI", "WMB", "EPD", "ET", "MPLX", "LNG", "OKE", "TRGP",
        "BTU", "ARCH", "AMR", "CEIX", "ARLP",
    ],
    "Utilities": [
        "NEE", "D", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "ES", "WEC",
        "CMS", "NI", "LNT", "PPL", "AEE", "DTE", "PNW", "EVRG", "ETR", "CNP",
        "AWK", "WTR", "SWX", "NJR", "SR",
    ],
    "Real_Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "SPG", "EQR", "AVB",
        "VTR", "WELL", "PEAK", "SBAC", "AMH", "INVH", "ELS", "SUI", "UDR",
        "CPT", "MAA", "ESS", "NLY", "AGNC", "STWD", "BXMT",
        "BXP", "VNO", "SLG", "ARE", "LTC", "NHI", "SBRA",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "SCCO", "NUE", "STLD",
        "AA", "ALB", "MP", "PPG", "RPM", "AXTA", "CC", "HUN",
        "BLL", "SON", "ATR", "IP", "PKG", "SEE", "GPK",
        "VMC", "MLM", "CRH", "EXP", "SUM", "WRK", "OLN", "CE",
    ],
    "Canada": [
        "SHOP.TO", "RY.TO", "TD.TO", "ENB.TO", "CNR.TO", "BNS.TO", "BMO.TO",
        "SU.TO", "ABX.TO", "CP.TO", "CSU.TO", "MFC.TO", "TRP.TO", "WCN.TO",
        "AEM.TO", "NTR.TO", "CNQ.TO", "CVE.TO", "T.TO", "BCE.TO",
        "MG.TO", "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "WN.TO",
        "GIB-A.TO", "CGI.TO", "OTEX.TO", "CAE.TO", "WSP.TO",
        "AQN.TO", "FTS.TO", "EMA.TO", "CU.TO", "BAM.TO",
        "WPM.TO", "K.TO", "AEM.TO", "ELD.TO", "FR.TO",
    ],
    "ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "EEM",
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE",
        "XLC", "XLY", "GLD", "SLV", "GDX", "GDXJ",
        "TLT", "IEF", "AGG", "LQD", "HYG",
        "OIH", "XOP", "KRE", "KBE",
        "SMH", "SOXX", "IGV", "SKYY", "IBB", "XBI",
    ],
}

# ── Flat reverse-lookup: ticker → sector ─────────────────────────────────────
_TICKER_SECTOR: Dict[str, str] = {}
for _sector, _tickers in _UNIVERSE.items():
    for _t in _tickers:
        if _t not in _TICKER_SECTOR:
            _TICKER_SECTOR[_t] = _sector


def get_ticker_sector(ticker: str) -> str:
    """Return sector name for *ticker*; 'Unknown' if not in universe."""
    return _TICKER_SECTOR.get(ticker.upper(), "Unknown")


def get_all_universe_tickers() -> List[str]:
    """Return de-duplicated flat list of all universe tickers."""
    seen = set()
    out = []
    for tickers in _UNIVERSE.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


class UniverseAgent:
    """
    Pre-screens the full universe down to a sector-balanced candidate list
    for the expensive stages (TimesFM, Chronos, ML).
    """

    def __init__(self, config: dict):
        self.config          = config
        u_cfg                = config.get("universe", {})
        self.max_prescreen   = int(u_cfg.get("max_prescreen",   150))
        self.min_avg_volume  = float(u_cfg.get("min_avg_volume", 500_000))
        self.min_price       = float(u_cfg.get("min_price",      1.0))
        self.momentum_window = int(u_cfg.get("momentum_window",  20))
        self.per_sector_cap  = int(u_cfg.get("per_sector_cap",   15))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> Optional[Dict[str, dict]]:
        """
        Returns {ticker -> {sector, momentum, avg_volume}} for up to
        *max_prescreen* candidates, or None if user set a manual override.
        """
        manual_tickers = get_all_tickers(self.config)
        if len(manual_tickers) < 30:
            logger.info(
                "Manual ticker override detected (%d tickers) — skipping universe screen",
                len(manual_tickers),
            )
            return None

        all_tickers = get_all_universe_tickers()
        logger.info(
            "Universe: %d total tickers across %d sectors",
            len(all_tickers), len(_UNIVERSE),
        )

        # Fast bulk fetch for liquidity + momentum screening
        screened = self._bulk_screen(all_tickers)
        if not screened:
            logger.warning("Universe bulk screen failed — returning None (use config tickers)")
            return None

        # Sector-balanced top-K selection
        candidates = self._select_candidates(screened)
        logger.info(
            "Universe pre-screen complete: %d candidates from %d sectors",
            len(candidates), len({v["sector"] for v in candidates.values()}),
        )
        return candidates

    # ── Bulk screening ────────────────────────────────────────────────────────

    def _bulk_screen(self, tickers: List[str]) -> Dict[str, dict]:
        """Download recent prices in bulk; compute momentum + volume filter."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not available for universe screening")
            return {}

        logger.info("Bulk downloading %d-day history for %d tickers …",
                    self.momentum_window + 5, len(tickers))

        # yfinance batch download — significantly faster than per-ticker
        period = f"{max(1, (self.momentum_window + 5) // 20 + 1)}mo"
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.error("Bulk download failed: %s", exc)
            return {}

        results: Dict[str, dict] = {}
        today_idx = -1   # last available row

        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    df = raw[ticker].dropna(how="all")
                else:
                    # Single ticker case
                    df = raw.dropna(how="all")

                if df.empty or len(df) < self.momentum_window:
                    continue

                close  = df["Close"].dropna()
                volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

                if close.empty:
                    continue

                last_price = float(close.iloc[-1])
                if last_price < self.min_price:
                    continue

                avg_vol = float(volume.tail(self.momentum_window).mean()) if not volume.empty else 0.0
                if avg_vol < self.min_avg_volume:
                    continue

                # 20-day momentum
                momentum = float(
                    close.iloc[-1] / close.iloc[-self.momentum_window] - 1
                ) if len(close) >= self.momentum_window else 0.0

                results[ticker] = {
                    "sector":      get_ticker_sector(ticker),
                    "last_price":  round(last_price, 2),
                    "avg_volume":  round(avg_vol, 0),
                    "momentum":    round(momentum, 4),
                }
            except Exception:
                continue

        logger.info(
            "Bulk screen: %d/%d tickers passed price/volume filters",
            len(results), len(tickers),
        )
        return results

    # ── Candidate selection ───────────────────────────────────────────────────

    def _select_candidates(self, screened: Dict[str, dict]) -> Dict[str, dict]:
        """
        Select up to *per_sector_cap* tickers per sector by momentum score,
        total capped at *max_prescreen*.
        """
        # Group by sector
        by_sector: Dict[str, List] = {}
        for ticker, info in screened.items():
            sector = info["sector"]
            by_sector.setdefault(sector, []).append((ticker, info["momentum"], info))

        # Sort each sector bucket by momentum descending
        for sector in by_sector:
            by_sector[sector].sort(key=lambda x: x[1], reverse=True)

        selected: Dict[str, dict] = {}
        # Round-robin across sectors to ensure diversity
        sector_iters = {s: iter(items) for s, items in by_sector.items()}
        sectors_active = list(sector_iters.keys())

        while len(selected) < self.max_prescreen and sectors_active:
            next_active = []
            for sector in sectors_active:
                try:
                    ticker, momentum, info = next(sector_iters[sector])
                    # Respect per-sector cap
                    sector_count = sum(
                        1 for v in selected.values() if v["sector"] == sector
                    )
                    if sector_count < self.per_sector_cap:
                        selected[ticker] = info
                        next_active.append(sector)
                    else:
                        pass  # sector capped; skip but sector may still have remaining
                except StopIteration:
                    pass  # sector exhausted
            if not next_active:
                break
            sectors_active = next_active
            if len(selected) >= self.max_prescreen:
                break

        return dict(list(selected.items())[: self.max_prescreen])
