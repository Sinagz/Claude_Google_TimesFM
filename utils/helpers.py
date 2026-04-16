"""
General-purpose helpers: logging, config loading, caching, I/O.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger that writes to stdout *and* logs/<name>.log."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid adding duplicate handlers
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False     # prevent double-printing via root logger

    fmt = logging.Formatter(
        "%(asctime)s [%(name)-22s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/{name}.log", encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and return the YAML config as a plain dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all required output directories from config."""
    paths = [
        config["data"]["save_path"],
        os.path.dirname(config["output_path"]) or ".",
        "logs",
        "models/cache",
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def is_cache_fresh(filepath: str, max_age_hours: float) -> bool:
    """Return True if *filepath* exists and is younger than *max_age_hours*."""
    if not os.path.exists(filepath):
        return False
    age = (datetime.now().timestamp() - os.path.getmtime(filepath)) / 3600
    return age < max_age_hours


# ── Date utilities ────────────────────────────────────────────────────────────

def date_n_days_ago(n: int) -> str:
    """Return ISO date string for N calendar days ago."""
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(data: Any, filepath: str) -> None:
    """Pretty-print *data* as JSON to *filepath*, creating dirs as needed."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Score normalisation ───────────────────────────────────────────────────────

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a dict of floats to [0, 1]."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def get_all_tickers(config: Dict[str, Any]):
    """Return flat list of all configured tickers."""
    return config["tickers"]["us"] + config["tickers"]["canada"]
