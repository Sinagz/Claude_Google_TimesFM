"""
Thin, retry-wrapped clients for every external API used by the system.

Rate-limit awareness
────────────────────
Alpha Vantage free  : 25 req / day  → used only for supplementary calls
yfinance            : no hard limit  → primary OHLCV source
Finnhub free        : 60 req / min  → used for company profile / fundamentals
NewsAPI.org         : 100 req / day (free) / 500 req / day (developer)
NewsAPI.ai          : key-dependent → used as secondary news source
"""

import time
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.helpers import setup_logger

logger = setup_logger("api_clients")

# ── Retry decorator shared by all HTTP methods ────────────────────────────────

_RETRY = retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)


def _get(url: str, params: dict, timeout: int = 20) -> Optional[Dict]:
    """Shared GET helper with logging and error guard."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        logger.warning("HTTP %s for %s — %s", exc.response.status_code, url, exc)
    except Exception as exc:
        logger.warning("Request failed for %s — %s", url, exc)
    return None


# ── Alpha Vantage ─────────────────────────────────────────────────────────────

class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._call_count = 0
        # Free tier: 25 req/day — track calls in this session
        self._DAILY_LIMIT = 25

    def _params(self, extra: dict) -> dict:
        return {"apikey": self.api_key, **extra}

    def _call(self, params: dict) -> Optional[Dict]:
        if self._call_count >= self._DAILY_LIMIT:
            logger.warning("Alpha Vantage daily limit reached, skipping call.")
            return None
        data = _get(self.BASE, params)
        self._call_count += 1
        # Respect 5 req/min rate limit on free tier
        time.sleep(12)
        return data

    def get_daily_adjusted(self, symbol: str, outputsize: str = "full") -> Optional[Dict]:
        """Daily adjusted OHLCV for *symbol*."""
        return self._call(self._params({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
        }))

    def get_news_sentiment(self, tickers: List[str], limit: int = 50) -> Optional[Dict]:
        """Alpha Vantage News & Sentiment endpoint."""
        ticker_str = ",".join(tickers)
        return self._call(self._params({
            "function": "NEWS_SENTIMENT",
            "tickers": ticker_str,
            "limit": limit,
        }))

    def get_overview(self, symbol: str) -> Optional[Dict]:
        """Company overview / fundamentals."""
        return self._call(self._params({"function": "OVERVIEW", "symbol": symbol}))


# ── Finnhub ───────────────────────────────────────────────────────────────────

class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _params(self, extra: dict) -> dict:
        return {"token": self.api_key, **extra}

    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> Optional[List]:
        data = _get(f"{self.BASE}/company-news", self._params({
            "symbol": symbol, "from": from_date, "to": to_date,
        }))
        return data if isinstance(data, list) else None

    def get_basic_financials(self, symbol: str) -> Optional[Dict]:
        return _get(f"{self.BASE}/stock/metric", self._params({
            "symbol": symbol, "metric": "all",
        }))

    def get_recommendation_trends(self, symbol: str) -> Optional[List]:
        data = _get(f"{self.BASE}/stock/recommendation", self._params({"symbol": symbol}))
        return data if isinstance(data, list) else None


# ── NewsAPI.org ───────────────────────────────────────────────────────────────

class NewsAPIClient:
    BASE = "https://newsapi.org/v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._call_count = 0
        self._DAILY_LIMIT = 100  # free-tier ceiling

    def get_everything(
        self,
        query: str,
        from_date: Optional[str] = None,
        language: str = "en",
        page_size: int = 20,
    ) -> Optional[Dict]:
        if self._call_count >= self._DAILY_LIMIT:
            logger.warning("NewsAPI.org daily limit reached.")
            return None
        data = _get(f"{self.BASE}/everything", {
            "apiKey": self.api_key,
            "q": query,
            "from": from_date or "",
            "language": language,
            "pageSize": page_size,
            "sortBy": "relevancy",
        })
        if data:
            self._call_count += 1
        return data


# ── NewsAPI.ai ────────────────────────────────────────────────────────────────

class NewsAIClient:
    """Client for newsapi.ai (EventRegistry API)."""
    BASE = "https://eventregistry.org/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_articles(self, keyword: str, max_items: int = 20) -> Optional[Dict]:
        payload = {
            "action": "getArticles",
            "keyword": keyword,
            "articlesPage": 1,
            "articlesCount": max_items,
            "articlesSortBy": "date",
            "resultType": "articles",
            "dataType": ["news"],
            "apiKey": self.api_key,
        }
        try:
            resp = requests.post(
                f"{self.BASE}/article/getArticles",
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("NewsAPI.ai request failed: %s", exc)
            return None


# ── yfinance wrapper ──────────────────────────────────────────────────────────

class YFinanceClient:
    """Thin wrapper so callers don't import yfinance directly."""

    @staticmethod
    def get_history(symbol: str, period: str = "2y", interval: str = "1d"):
        """Return a pandas DataFrame of OHLCV data."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                logger.warning("yfinance returned empty frame for %s", symbol)
                return None
            return df
        except Exception as exc:
            logger.error("yfinance error for %s: %s", symbol, exc)
            return None
