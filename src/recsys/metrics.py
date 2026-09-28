"""Top-K ranking metrics. All computed per user so they can be bootstrapped over users."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config


def top_k(scores: np.ndarray, exclude: sp.csr_matrix, k: int = config.K) -> np.ndarray:
    """Indices of the k best-scoring items per user, best first, never an already-seen item."""
    s = scores.astype(np.float32, copy=True)
    e = exclude.tocoo()
    s[e.row, e.col] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def per_user(recs: np.ndarray, relevant: sp.csr_matrix, k: int = config.K) -> dict[str, np.ndarray]:
    """precision@k, recall@k, hit-rate@k and binary NDCG@k for every user (rows aligned with `relevant`)."""
    rel = relevant.tocsr()
    n = recs.shape[0]
    hits = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        rel_items = set(rel.indices[rel.indptr[i]: rel.indptr[i + 1]])
        hits[i] = [r in rel_items for r in recs[i]]
    n_rel = np.diff(rel.indptr)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discount
    ideal = np.array([discount[: min(int(m), k)].sum() for m in n_rel])
    return {"precision": hits.sum(1) / k, "recall": hits.sum(1) / np.maximum(n_rel, 1),
            "hit": (hits.sum(1) > 0).astype(np.float32), "ndcg": dcg / np.maximum(ideal, 1e-9)}


def catalogue_stats(recs: np.ndarray, popularity: np.ndarray, n_items: int) -> dict[str, float]:
    """Coverage: share of the catalogue that is ever recommended. Novelty: mean -log2 of a movie's share
    of likes, so higher = more niche recommendations."""
    share = popularity / max(1.0, popularity.sum())
    novelty = -np.log2(np.maximum(share[recs], 1e-9))
    return {"coverage": len(np.unique(recs)) / n_items, "novelty": float(novelty.mean()),
            "mean_popularity_rank": float(np.mean(np.argsort(np.argsort(-popularity))[recs]))}


def bootstrap_mean(values: np.ndarray, n: int = 1000, seed: int = config.SEED) -> tuple[float, float]:
    """95% interval of the mean, resampling users."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = config.SEED) -> tuple[float, float]:
    """95% interval of mean(a - b) for two models scored on the same users (paired)."""
    return bootstrap_mean(a - b, n, seed)
