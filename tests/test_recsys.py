"""Offline tests on small synthetic data (no MovieLens download, no trained artifacts needed)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys import data, metrics
from recsys import models as M
from recsys.explain import because_you_liked, shared_genres


def _ratings(n_users=60, per_user=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for t, i in enumerate(rng.choice(40, per_user, replace=False)):
            rows.append((u, int(i), int(rng.integers(1, 6)), 1000 + t))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def _clustered(n_per=80, n_items=40, per_user=12, seed=0):
    """Two taste groups: group A likes items 0-19, group B likes items 20-39, plus a global hit (item 0)."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for u in range(2 * n_per):
        pool = np.arange(20) if u < n_per else np.arange(20, 40)
        for i in rng.choice(pool, per_user, replace=False):
            rows.append(u)
            cols.append(int(i))
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(2 * n_per, n_items))


# ---------------------------------------------------------------- data
def test_chronological_split_is_per_user_and_ordered():
    r = _ratings()
    lab = data.chronological_split(r)
    assert set(lab) == {"train", "val", "test"}
    for _, g in r.assign(lab=lab).groupby("user"):
        g = g.sort_values("ts")
        order = {"train": 0, "val": 1, "test": 2}
        assert list(g.lab.map(order)) == sorted(g.lab.map(order))  # never trains on the user's future
        assert (g.lab == "train").mean() == pytest.approx(0.7, abs=0.05)


def test_to_csr_positive_only_keeps_liked():
    r = pd.DataFrame({"user": [0, 0, 1], "item": [0, 1, 1], "rating": [5, 2, 4], "ts": [1, 2, 3]})
    assert data.to_csr(r, 2, 2).nnz == 2
    assert data.to_csr(r, 2, 2, positive_only=False).nnz == 3


# ---------------------------------------------------------------- metrics
def test_top_k_never_returns_seen_items():
    scores = np.array([[0.9, 0.8, 0.7, 0.1]])
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0]]))
    assert list(metrics.top_k(scores, seen, 2)[0]) == [1, 2]


def test_ranking_metrics_hand_computed():
    recs = np.array([[3, 1, 2]])
    rel = sp.csr_matrix(np.array([[0, 1, 0, 1]]))  # liked items 1 and 3
    m = metrics.per_user(recs, rel, k=3)
    assert m["precision"][0] == pytest.approx(2 / 3)
    assert m["recall"][0] == pytest.approx(1.0)
    assert m["hit"][0] == 1
    # hits at ranks 1 and 2 = ideal ordering of 2 relevant items -> NDCG 1
    assert m["ndcg"][0] == pytest.approx(1.0)
    late = metrics.per_user(np.array([[0, 2, 3]]), rel, k=3)  # only hit at rank 3
    assert late["ndcg"][0] == pytest.approx((1 / np.log2(4)) / (1 + 1 / np.log2(3)))


def test_bootstrap_interval_brackets_mean():
    v = np.random.default_rng(1).random(400)
    lo, hi = metrics.bootstrap_mean(v, n=200)
    assert lo <= v.mean() <= hi


# ---------------------------------------------------------------- models
def _split_hist(X, hold=3, seed=0):
    """Hide `hold` liked items per user; return (history, hidden)."""
    rng = np.random.default_rng(seed)
    hist, hid = X.tolil(), sp.lil_matrix(X.shape, dtype=np.float32)
    for u in range(X.shape[0]):
        items = X.indices[X.indptr[u]: X.indptr[u + 1]]
        for i in rng.choice(items, hold, replace=False):
            hist[u, i] = 0
            hid[u, i] = 1
    return hist.tocsr(), hid.tocsr()


@pytest.mark.parametrize("model", [M.ItemKNN(k=10, shrink=1.0), M.ALS(factors=8, reg=0.1, alpha=10, iters=6)])
def test_collaborative_models_beat_popularity_on_planted_taste_groups(model):
    X = _clustered()
    hist, hidden = _split_hist(X)
    pop = M.Popularity().fit(hist)
    model.fit(hist)
    seen = hist

    def ndcg(m):
        recs = metrics.top_k(m.score(hist), seen, 5)
        return metrics.per_user(recs, hidden, 5)["ndcg"].mean()

    assert ndcg(model) > ndcg(pop) + 0.1


def test_als_score_splits_exactly_over_liked_movies():
    X = _clustered()
    als = M.ALS(factors=8, reg=0.1, alpha=10, iters=5).fit(X)
    hist_items = X.indices[X.indptr[3]: X.indptr[4]]
    item = 7
    score = als.score(X[3])[0, item]
    assert als.contributions(hist_items, item).sum() == pytest.approx(score, rel=1e-4, abs=1e-5)


def test_als_fold_in_is_deterministic_and_history_only():
    X = _clustered()
    als = M.ALS(factors=8, iters=4).fit(X)
    a, b = als.fold_in(X[5]), als.fold_in(X[5])
    assert np.allclose(a, b)
    assert np.allclose(als.fold_in(sp.csr_matrix((1, X.shape[1]), dtype=np.float32)), 0)  # no history -> no signal


def test_content_model_scores_an_item_with_no_interactions():
    movies = pd.DataFrame({"title": ["Star Trek", "Star Trek II", "Romcom A", "Romcom B"],
                           "year": [1979, 1982, 1999, 2001],
                           "genres": [["Sci-Fi"], ["Sci-Fi"], ["Romance"], ["Romance"]]})
    c = M.Content(movies)
    hist = sp.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))  # liked the first film only
    s = c.score(hist)[0]
    assert s[1] > s[2] and s[1] > s[3]  # the sequel, which nobody has rated, ranks above the romcoms


def test_hybrid_is_weighted_sum_of_standardised_scores():
    X = _clustered()
    a, k = M.ALS(factors=8, iters=3).fit(X), M.ItemKNN(k=10).fit(X)
    h = M.Hybrid([(a, 1.0), (k, 0.5)])
    expect = M.zscore_rows(a.score(X)) + 0.5 * M.zscore_rows(k.score(X))
    assert np.allclose(h.score(X), expect, atol=1e-5)


# ---------------------------------------------------------------- explanations
def test_because_you_liked_returns_shares_that_sum_to_one_and_are_ranked():
    X = _clustered()
    als = M.ALS(factors=8, iters=4).fit(X)
    hist_items = X.indices[X.indptr[0]: X.indptr[1]]
    out = because_you_liked(als, hist_items, 5, k=len(hist_items))
    shares = [s for _, s in out]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0)
    assert because_you_liked(als, np.array([], dtype=int), 5) == []


def test_shared_genres_counts_overlap_with_liked_movies():
    movies = pd.DataFrame({"genres": [["Drama"], ["Drama", "Crime"], ["Crime"], ["Comedy"]]})
    assert shared_genres(movies, np.array([0, 1]), 2) == [("Crime", 1)]
