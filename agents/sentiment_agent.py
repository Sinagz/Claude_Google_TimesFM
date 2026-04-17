"""
Sentiment Analysis Agent
─────────────────────────
Fetches recent news for each ticker and scores sentiment using FinBERT
(primary) or VADER (fallback).

Decay weighting
  Articles are weighted by recency using exponential decay:
    weight = exp(-ln(2) / HALF_LIFE_DAYS × days_ago)
  where HALF_LIFE_DAYS = 3.5  (an article 3.5 days old carries half the
  weight of today's article).  This ensures recent bullish/bearish signals
  dominate over stale ones.

Output: {ticker -> score} in [-1, +1]
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.api_clients import (
    AlphaVantageClient,
    FinnhubClient,
    NewsAIClient,
    NewsAPIClient,
)
from utils.helpers import date_n_days_ago, setup_logger

logger = setup_logger("sentiment_agent")

_FINBERT_MAP        = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
_ARTICLES_PER_TICKER = 10
_NEWS_DAYS           = 7
_DECAY_HALF_LIFE     = 3.5   # days — half-life for recency weighting

# Article = (text_snippet, days_ago)
Article = Tuple[str, int]


def _decay_weight(days_ago: int) -> float:
    """Exponential decay: weight halves every _DECAY_HALF_LIFE days."""
    rate = math.log(2) / _DECAY_HALF_LIFE
    return math.exp(-rate * max(0, days_ago))


def _parse_days_ago(dt_str: str, fmt: Optional[str] = None) -> int:
    """Parse a datetime string and return how many days ago it was."""
    today = datetime.now()
    try:
        if fmt:
            pub = datetime.strptime(dt_str, fmt)
        else:
            # ISO 8601 with possible trailing Z
            pub = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            pub = pub.replace(tzinfo=None)
        return max(0, (today - pub).days)
    except Exception:
        return _NEWS_DAYS // 2   # default: mid-range age


class SentimentAgent:
    def __init__(self, config: dict):
        self.config  = config
        self.newsapi = NewsAPIClient(config["newsapi_org_key"])
        self.newsai  = NewsAIClient(config["newsapi_ai_key"])
        self.av      = AlphaVantageClient(config["alphavantage_api_key"])
        self.fh      = FinnhubClient(config["finnhub_api_key"])
        self._finbert     = None
        self._vader       = None
        self._use_finbert = True

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tickers: List[str]) -> Dict[str, float]:
        """Returns {ticker -> sentiment_score} in [-1, +1] with decay weighting."""
        self._init_scorer()
        results: Dict[str, float] = {}
        from_date = date_n_days_ago(_NEWS_DAYS)

        logger.info("Fetching & scoring news for %d tickers …", len(tickers))

        for ticker in tickers:
            articles = self._fetch_news(ticker, from_date)
            if articles:
                score = self._score_articles(articles)
            else:
                logger.debug("No news for %s — neutral score", ticker)
                score = 0.0
            results[ticker] = round(score, 4)
            logger.debug(
                "%s  sentiment=%.3f  articles=%d", ticker, score, len(articles)
            )

        logger.info("Sentiment scoring complete")
        return results

    # ── Scorer initialisation ─────────────────────────────────────────────────

    def _init_scorer(self) -> None:
        if self._finbert is not None or self._vader is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device_idx = 0 if torch.cuda.is_available() else -1
            logger.info("Loading FinBERT …")
            self._finbert = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=device_idx,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded")
            self._use_finbert = True
        except Exception as exc:
            logger.warning("FinBERT unavailable (%s) — falling back to VADER", exc)
            self._init_vader()
            self._use_finbert = False

    def _init_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except Exception:
            try:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except Exception as exc2:
                logger.error("VADER unavailable: %s — scores will be 0", exc2)

    # ── News fetching ─────────────────────────────────────────────────────────

    def _fetch_news(self, ticker: str, from_date: str) -> List[Article]:
        """Try each API in order; return up to _ARTICLES_PER_TICKER articles."""
        query = ticker.replace(".TO", "").replace(".V", "")
        articles: List[Article] = []

        articles = self._from_newsapi_org(query, from_date)
        if len(articles) >= _ARTICLES_PER_TICKER:
            return articles[:_ARTICLES_PER_TICKER]

        articles += self._from_newsapi_ai(query)
        if len(articles) >= _ARTICLES_PER_TICKER:
            return articles[:_ARTICLES_PER_TICKER]

        articles += self._from_alpha_vantage(ticker)
        if len(articles) >= _ARTICLES_PER_TICKER:
            return articles[:_ARTICLES_PER_TICKER]

        articles += self._from_finnhub(query, from_date)
        return articles[:_ARTICLES_PER_TICKER]

    def _from_newsapi_org(self, query: str, from_date: str) -> List[Article]:
        try:
            data = self.newsapi.get_everything(
                query=query, from_date=from_date, page_size=_ARTICLES_PER_TICKER,
            )
            if not data or data.get("status") != "ok":
                return []
            articles = []
            for a in data.get("articles", []):
                title = (a.get("title") or "").strip()
                desc  = (a.get("description") or "")[:200]
                text  = f"{title}. {desc}".strip(" .")
                if not text:
                    continue
                days_ago = _parse_days_ago(a.get("publishedAt", ""))
                articles.append((text, days_ago))
            return articles
        except Exception as exc:
            logger.debug("NewsAPI.org error %s: %s", query, exc)
            return []

    def _from_newsapi_ai(self, query: str) -> List[Article]:
        try:
            data = self.newsai.get_articles(keyword=query, max_items=_ARTICLES_PER_TICKER)
            if not data:
                return []
            articles = []
            for a in data.get("articles", {}).get("results", []):
                title = (a.get("title") or "").strip()
                body  = (a.get("body") or "")[:200]
                text  = f"{title}. {body}".strip(" .")
                if not text:
                    continue
                days_ago = _parse_days_ago(a.get("dateTime", ""))
                articles.append((text, days_ago))
            return articles
        except Exception as exc:
            logger.debug("NewsAPI.ai error %s: %s", query, exc)
            return []

    def _from_alpha_vantage(self, ticker: str) -> List[Article]:
        try:
            data = self.av.get_news_sentiment([ticker], limit=_ARTICLES_PER_TICKER)
            if not data:
                return []
            articles = []
            for item in data.get("feed", []):
                title   = (item.get("title") or "").strip()
                summary = (item.get("summary") or "")[:200]
                text    = f"{title}. {summary}".strip(" .")
                if not text:
                    continue
                # AV format: "20240416T120000"
                days_ago = _parse_days_ago(
                    item.get("time_published", ""), fmt="%Y%m%dT%H%M%S"
                )
                articles.append((text, days_ago))
            return articles
        except Exception as exc:
            logger.debug("AV news error %s: %s", ticker, exc)
            return []

    def _from_finnhub(self, symbol: str, from_date: str) -> List[Article]:
        try:
            to_date  = datetime.now().strftime("%Y-%m-%d")
            raw_list = self.fh.get_company_news(symbol, from_date, to_date)
            if not raw_list:
                return []
            articles = []
            for a in raw_list[:_ARTICLES_PER_TICKER]:
                headline = (a.get("headline") or "").strip()
                summary  = (a.get("summary") or "")[:200]
                text     = f"{headline}. {summary}".strip(" .")
                if not text:
                    continue
                # Finnhub datetime is a UNIX timestamp
                ts = a.get("datetime", 0)
                try:
                    pub = datetime.fromtimestamp(int(ts))
                    days_ago = max(0, (datetime.now() - pub).days)
                except Exception:
                    days_ago = _NEWS_DAYS // 2
                articles.append((text, days_ago))
            return articles
        except Exception as exc:
            logger.debug("Finnhub news error %s: %s", symbol, exc)
            return []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_articles(self, articles: List[Article]) -> float:
        """Exponentially decay-weighted average of per-article sentiment scores."""
        if not articles:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for text, days_ago in articles:
            score  = self._score_one(text)
            weight = _decay_weight(days_ago)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return float(weighted_sum / total_weight)

    def _score_one(self, text: str) -> float:
        """Score a single text. Returns value in [-1, +1]."""
        if self._use_finbert and self._finbert:
            try:
                result = self._finbert(text[:512])[0]
                label  = result["label"].lower()
                conf   = float(result["score"])
                return _FINBERT_MAP.get(label, 0.0) * conf
            except Exception:
                pass

        if self._vader:
            try:
                scores = self._vader.polarity_scores(text)
                return float(scores["compound"])
            except Exception:
                pass

        return 0.0
