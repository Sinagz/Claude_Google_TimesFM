"""
Clustering Agent — Diversification Engine
──────────────────────────────────────────
Groups stocks by return-profile similarity so the ranking agent can
enforce intra-cluster diversity and avoid correlated bets.

Features used per ticker (from featured_data):
  • volatility_21d   — short-term volatility regime
  • momentum_21d     — recent price momentum
  • return_21d       — trailing 21-day log return
  • trend_strength   — position relative to 50-day SMA

Cluster labels
  The agent runs KMeans(n_clusters=N) and assigns each ticker to a cluster.
  N is capped at min(n_clusters_config, n_tickers // 3) so small universes
  don't blow up.

Output
  {
    "cluster_map":     {ticker: cluster_id},
    "cluster_members": {cluster_id: [ticker, ...]},
    "n_clusters":      int,
    "features_used":   [...],
  }
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import setup_logger

logger = setup_logger("clustering_agent")

_CLUSTER_FEATURES = [
    "volatility_21d",
    "momentum_21d",
    "return_21d",
    "trend_strength",
]


class ClusteringAgent:
    def __init__(self, config: dict):
        clust = config.get("clustering", {})
        self.n_clusters = int(clust.get("n_clusters", 8))
        self.random_state = int(clust.get("random_state", 42))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, featured_data: Dict[str, pd.DataFrame]) -> dict:
        """
        Parameters
        ----------
        featured_data : {ticker -> enriched DataFrame from FeatureAgent}

        Returns
        -------
        dict with cluster_map, cluster_members, n_clusters, features_used
        """
        if not featured_data:
            return self._empty_result()

        feature_matrix, valid_tickers = self._build_feature_matrix(featured_data)

        if len(valid_tickers) < 3:
            logger.warning("Too few tickers for clustering (%d); skipping", len(valid_tickers))
            # Each ticker is its own cluster
            cluster_map = {t: i for i, t in enumerate(valid_tickers)}
            cluster_members = {i: [t] for i, t in enumerate(valid_tickers)}
            return {
                "cluster_map":     cluster_map,
                "cluster_members": cluster_members,
                "n_clusters":      len(valid_tickers),
                "features_used":   _CLUSTER_FEATURES,
            }

        n = min(self.n_clusters, max(2, len(valid_tickers) // 3))
        labels = self._fit_kmeans(feature_matrix, n)

        cluster_map: Dict[str, int] = {}
        cluster_members: Dict[int, List[str]] = {}
        for ticker, label in zip(valid_tickers, labels):
            cluster_map[ticker] = int(label)
            cluster_members.setdefault(int(label), []).append(ticker)

        logger.info(
            "Clustering complete: %d tickers → %d clusters",
            len(valid_tickers), n,
        )
        for cid, members in sorted(cluster_members.items()):
            logger.debug("  Cluster %d: %s", cid, members)

        return {
            "cluster_map":     cluster_map,
            "cluster_members": cluster_members,
            "n_clusters":      n,
            "features_used":   _CLUSTER_FEATURES,
        }

    # ── Feature matrix ────────────────────────────────────────────────────────

    @staticmethod
    def _build_feature_matrix(
        featured_data: Dict[str, pd.DataFrame],
    ) -> tuple:
        rows = []
        tickers = []
        for ticker, df in featured_data.items():
            if df is None or df.empty:
                continue
            row = []
            ok = True
            for col in _CLUSTER_FEATURES:
                if col in df.columns:
                    val = df[col].dropna()
                    row.append(float(val.iloc[-1]) if not val.empty else 0.0)
                else:
                    row.append(0.0)
            rows.append(row)
            tickers.append(ticker)

        if not rows:
            return np.empty((0, len(_CLUSTER_FEATURES))), []

        X = np.array(rows, dtype=float)
        # Standardise each column
        means = np.nanmean(X, axis=0)
        stds  = np.nanstd(X, axis=0)
        stds[stds == 0] = 1.0
        X = (X - means) / stds
        X = np.nan_to_num(X, nan=0.0)
        return X, tickers

    # ── KMeans ────────────────────────────────────────────────────────────────

    def _fit_kmeans(self, X: np.ndarray, n: int) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=n,
                n_init=10,
                random_state=self.random_state,
                max_iter=300,
            )
            return km.fit_predict(X)
        except Exception as exc:
            logger.error("KMeans failed: %s — using fallback", exc)
            return self._fallback_clusters(X, n)

    @staticmethod
    def _fallback_clusters(X: np.ndarray, n: int) -> np.ndarray:
        """Sort by first feature (volatility), split into n equal buckets."""
        order = np.argsort(X[:, 0])
        labels = np.zeros(len(X), dtype=int)
        bucket_size = max(1, len(X) // n)
        for i, idx in enumerate(order):
            labels[idx] = min(i // bucket_size, n - 1)
        return labels

    @staticmethod
    def _empty_result() -> dict:
        return {
            "cluster_map":     {},
            "cluster_members": {},
            "n_clusters":      0,
            "features_used":   _CLUSTER_FEATURES,
        }
