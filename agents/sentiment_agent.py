"""
Sentiment Analysis Agent
─────────────────────────
Fetches recent news for each ticker from multiple sources (with API rotation)
and scores each article's sentiment using:

  Primary  : FinBERT (ProsusAI/finbert) — fine-tuned on financial text
  Fallback : VADER  (rule-based, fast, no model download needed)

API rotation order (avoids exhausting any single quota):
  1. NewsAPI.org    (~100 req/day free)
  2. NewsAPI.ai     (EventRegistry — key-dependent)
  3. Alpha Vantage  News & Sentiment endpoint (~25 req/day budget shared)
  4. Finnhub        company-news endpoint (~60 req/min)

Returns {ticker -> sentiment_score} where scores are in [-1, +1].
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from utils.api_clients import (
    AlphaVantageClient,
    FinnhubClient,
    NewsAIClient,
    NewsAPIClient,
)
from utils.helpers import date_n_days_ago, setup_logger

logger = setup_logger("sentiment_agent")

# FinBERT label → numeric value
_FINBERT_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Number of articles to score per ticker
_ARTICLES_PER_TICKER = 10

# News look-back (days)
_NEWS_DAYS = 7


class SentimentAgent:
    def __init__(self, config: dict):
        self.config = config
        self.newsapi = NewsAPIClient(config["newsapi_org_key"])
        self.newsai  = NewsAIClient(config["newsapi_ai_key"])
        self.av      = AlphaVantageClient(config["alphavantage_api_key"])
        self.fh      = FinnhubClient(config["finnhub_api_key"])
        self._finbert   = None
        self._vader     = None
        self._use_finbert = True

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tickers: List[str]) -> Dict[str, float]:
        """
        Returns {ticker -> sentiment_score} in [-1, +1].
        Positive means bullish news; negative means bearish.
        """
        self._init_scorer()
        results: Dict[str, float] = {}
        from_date = date_n_days_ago(_NEWS_DAYS)

        logger.info("Fetching & scoring news for %d tickers …", len(tickers))

        for ticker in tickers:
            texts = self._fetch_news(ticker, from_date)
            if texts:
                score = self._score_texts(texts)
            else:
                logger.debug("No news found for %s — neutral score", ticker)
                score = 0.0
            results[ticker] = round(score, 4)
            logger.debug("%s sentiment=%.3f (%d articles)", ticker, score, len(texts))

        logger.info("Sentiment scoring complete")
        return results

    # ── Scorer initialisation ─────────────────────────────────────────────────

    def _init_scorer(self):
        """Try to load FinBERT; fall back to VADER if unavailable."""
        if self._finbert is not None or self._vader is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
            import torch

            device_idx = 0 if (torch.cuda.is_available()) else -1
            logger.info("Loading FinBERT sentiment model …")
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

    def _init_vader(self):
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
                logger.error("VADER also unavailable: %s — scores will be 0", exc2)

    # ── News fetching (with API rotation) ────────────────────────────────────

    def _fetch_news(self, ticker: str, from_date: str) -> List[str]:
        """Try each API in order; return as many article snippets as possible."""
        # Use plain company name for better search results
        query = ticker.replace(".TO", "").replace(".V", "")

        texts: List[str] = []

        # 1) NewsAPI.org
        texts = self._from_newsapi_org(query, from_date)
        if len(texts) >= _ARTICLES_PER_TICKER:
            return texts[:_ARTICLES_PER_TICKER]

        # 2) NewsAPI.ai
        texts += self._from_newsapi_ai(query)
        if len(texts) >= _ARTICLES_PER_TICKER:
            return texts[:_ARTICLES_PER_TICKER]

        # 3) Alpha Vantage news (only if budget remains)
        texts += self._from_alpha_vantage(ticker)
        if len(texts) >= _ARTICLES_PER_TICKER:
            return texts[:_ARTICLES_PER_TICKER]

        # 4) Finnhub
        texts += self._from_finnhub(query, from_date)

        return texts[:_ARTICLES_PER_TICKER]

    def _from_newsapi_org(self, query: str, from_date: str) -> List[str]:
        try:
            data = self.newsapi.get_everything(
                query=query,
                from_date=from_date,
                page_size=_ARTICLES_PER_TICKER,
            )
            if not data or data.get("status") != "ok":
                return []
            articles = data.get("articles", [])
            return self._extract_texts(articles, title_key="title", desc_key="description")
        except Exception as exc:
            logger.debug("NewsAPI.org error for %s: %s", query, exc)
            return []

    def _from_newsapi_ai(self, query: str) -> List[str]:
        try:
            data = self.newsai.get_articles(keyword=query, max_items=_ARTICLES_PER_TICKER)
            if not data:
                return []
            articles = data.get("articles", {}).get("results", [])
            texts = []
            for a in articles:
                title = a.get("title", "")
                body  = a.get("body", "")[:200]
                if title:
                    texts.append(f"{title}. {body}".strip())
            return texts
        except Exception as exc:
            logger.debug("NewsAPI.ai error for %s: %s", query, exc)
            return []

    def _from_alpha_vantage(self, ticker: str) -> List[str]:
        try:
            data = self.av.get_news_sentiment([ticker], limit=_ARTICLES_PER_TICKER)
            if not data:
                return []
            feed = data.get("feed", [])
            texts = []
            for item in feed:
                title   = item.get("title", "")
                summary = item.get("summary", "")[:200]
                if title:
                    texts.append(f"{title}. {summary}".strip())
            return texts
        except Exception as exc:
            logger.debug("AV news error for %s: %s", ticker, exc)
            return []

    def _from_finnhub(self, symbol: str, from_date: str) -> List[str]:
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            articles = self.fh.get_company_news(symbol, from_date, to_date)
            if not articles:
                return []
            texts = []
            for a in articles[:_ARTICLES_PER_TICKER]:
                headline = a.get("headline", "")
                summary  = a.get("summary", "")[:200]
                if headline:
                    texts.append(f"{headline}. {summary}".strip())
            return texts
        except Exception as exc:
            logger.debug("Finnhub news error for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _extract_texts(
        articles: list, title_key: str = "title", desc_key: str = "description"
    ) -> List[str]:
        texts = []
        for a in articles:
            title = a.get(title_key, "") or ""
            desc  = (a.get(desc_key, "") or "")[:200]
            combined = f"{title}. {desc}".strip(" .")
            if combined:
                texts.append(combined)
        return texts

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_texts(self, texts: List[str]) -> float:
        """Average sentiment across all article snippets."""
        if not texts:
            return 0.0
        scores = [self._score_one(t) for t in texts]
        return float(np.mean(scores))

    def _score_one(self, text: str) -> float:
        """Score a single text snippet. Returns value in [-1, +1]."""
        if self._use_finbert and self._finbert:
            try:
                result = self._finbert(text[:512])[0]
                label  = result["label"].lower()
                conf   = float(result["score"])
                return _FINBERT_MAP.get(label, 0.0) * conf
            except Exception:
                pass   # fall through to VADER

        if self._vader:
            try:
                scores = self._vader.polarity_scores(text)
                return float(scores["compound"])   # already in [-1, +1]
            except Exception:
                pass

        return 0.0   # last resort: neutral
