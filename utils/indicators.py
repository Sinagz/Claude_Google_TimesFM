"""
Technical indicator calculations on pandas DataFrames.

All functions accept a DataFrame with at least a 'Close' column
(plus 'High', 'Low', 'Volume' where noted) and return a new column
or a modified DataFrame.
"""

import numpy as np
import pandas as pd


# ── Returns & Volatility ─────────────────────────────────────────────────────

def add_returns(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """Log return over *window* days."""
    df[f"return_{window}d"] = np.log(df["Close"] / df["Close"].shift(window))
    return df


def add_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling annualised volatility (std of log returns × √252)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df[f"volatility_{window}d"] = log_ret.rolling(window).std() * np.sqrt(252)
    return df


def add_pct_return(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Simple percentage return over *window* days."""
    df[f"pct_return_{window}d"] = df["Close"].pct_change(window)
    return df


def add_drawdown(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling drawdown from the rolling *window*-day peak (negative fraction)."""
    rolling_max = df["Close"].rolling(window, min_periods=1).max()
    df[f"drawdown_{window}d"] = (df["Close"] - rolling_max) / rolling_max.replace(0, np.nan)
    return df


def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """(Close - SMA50) / SMA50 clipped to [-0.5, 0.5] — positive means above trend."""
    if "sma_50" not in df.columns:
        df = add_sma(df, 50)
    df["trend_strength"] = (
        (df["Close"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
    ).clip(-0.5, 0.5)
    return df


def add_sma_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """SMA10/SMA50 and SMA50/SMA200 as relative position indicators."""
    for w in (10, 50, 200):
        if f"sma_{w}" not in df.columns:
            df = add_sma(df, w)
    df["sma_ratio_10_50"]  = df["sma_10"]  / df["sma_50"].replace(0, np.nan)
    df["sma_ratio_50_200"] = df["sma_50"]  / df["sma_200"].replace(0, np.nan)
    return df


def add_momentum(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Price momentum: (Close / Close[window] - 1)."""
    df[f"momentum_{window}d"] = df["Close"] / df["Close"].shift(window) - 1
    return df


# ── Moving Averages ───────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df[f"sma_{window}"] = df["Close"].rolling(window).mean()
    return df


def add_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def add_golden_cross(df: pd.DataFrame) -> pd.DataFrame:
    """1 if SMA50 > SMA200 (bullish), -1 otherwise."""
    if "sma_50" not in df.columns:
        df = add_sma(df, 50)
    if "sma_200" not in df.columns:
        df = add_sma(df, 200)
    df["golden_cross"] = np.where(df["sma_50"] > df["sma_200"], 1.0, -1.0)
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index via Wilder's smoothing."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def add_bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Upper / lower Bollinger Bands and %B."""
    mid = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["bb_upper"] = mid + n_std * std
    df["bb_lower"] = mid - n_std * std
    df["bb_mid"] = mid
    band_width = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / band_width.replace(0, np.nan)
    return df


# ── ATR (Average True Range) ──────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Requires High, Low, Close columns."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{window}"] = tr.rolling(window).mean()
    return df


# ── Volume indicators ─────────────────────────────────────────────────────────

def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Current volume relative to rolling mean (Volume / SMA_volume)."""
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["obv"] = (direction * df["Volume"]).cumsum()
    return df


# ── Composite technical score ─────────────────────────────────────────────────

def compute_technical_score(df: pd.DataFrame) -> float:
    """
    Aggregate all indicators into a single bullishness score in [0, 1].
    Uses the most recent row of the fully-featured DataFrame.
    """
    row = df.dropna().iloc[-1] if not df.dropna().empty else df.iloc[-1]
    signals = []

    # RSI: oversold (<30) → bullish, overbought (>70) → bearish
    if "rsi_14" in row and not pd.isna(row["rsi_14"]):
        rsi = row["rsi_14"]
        signals.append(1.0 if rsi < 40 else (0.0 if rsi > 65 else 0.5))

    # MACD: positive histogram → bullish
    if "macd_hist" in row and not pd.isna(row["macd_hist"]):
        signals.append(1.0 if row["macd_hist"] > 0 else 0.0)

    # Golden cross
    if "golden_cross" in row and not pd.isna(row["golden_cross"]):
        signals.append(1.0 if row["golden_cross"] > 0 else 0.0)

    # Bollinger %B: below 0.2 → oversold → bullish
    if "bb_pct" in row and not pd.isna(row["bb_pct"]):
        bp = row["bb_pct"]
        signals.append(1.0 if bp < 0.25 else (0.0 if bp > 0.75 else 0.5))

    # Momentum
    if "momentum_21d" in row and not pd.isna(row["momentum_21d"]):
        mom = row["momentum_21d"]
        signals.append(min(max((mom + 0.1) / 0.2, 0.0), 1.0))

    return float(np.mean(signals)) if signals else 0.5


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators to *df* and return the enriched DataFrame."""
    df = df.copy()
    # Returns — log (model-ready) and pct (human-interpretable)
    df = add_returns(df, 1)
    df = add_returns(df, 5)
    df = add_returns(df, 21)
    df = add_pct_return(df, 5)
    df = add_pct_return(df, 21)
    # Volatility — 5d (short-term noise), 21d (monthly), 63d (quarterly)
    df = add_volatility(df, 5)
    df = add_volatility(df, 21)
    df = add_volatility(df, 63)
    # Momentum
    df = add_momentum(df, 21)
    df = add_momentum(df, 63)
    # Moving averages — 10 added for short-term ratio
    df = add_sma(df, 10)
    df = add_sma(df, 20)
    df = add_sma(df, 50)
    df = add_sma(df, 200)
    df = add_ema(df, 12)
    df = add_ema(df, 26)
    df = add_golden_cross(df)
    # Derived MA ratios and drawdown
    df = add_sma_ratios(df)
    df = add_drawdown(df, 21)
    df = add_trend_strength(df)
    # Oscillators
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_bollinger(df)
    if "High" in df.columns and "Low" in df.columns:
        df = add_atr(df)
    if "Volume" in df.columns:
        df = add_volume_ratio(df)
        df = add_obv(df)
    return df
