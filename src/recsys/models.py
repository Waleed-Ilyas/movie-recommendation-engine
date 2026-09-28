"""Recommenders. Every model turns a user's history into a score for every item:

    model.fit(train_positive_matrix)          # users x items, 1 = liked
    model.score(history_positive, signed)     # -> (n_users, n_items) scores

`score` works for any history, so a brand-new user (a handful of picks) goes through exactly the same
code path as a user seen in training. That is what makes the cold-start experiments honest."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import data


def zscore_rows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ popularity
class Popularity:
    """The same list for everyone: the movies most people liked. The baseline every other model must beat."""

    name = "Popularity"

    def fit(self, pos: sp.csr_matrix):
        self.pop = np.asarray(pos.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, hist: sp.csr_matrix, signed=None) -> np.ndarray:
        return np.tile(self.pop, (hist.shape[0], 1))


# ------------------------------------------------------------------ item-item neighbourhood
class ItemKNN:
    """Item-based collaborative filtering. Two movies are similar if the same people liked both (cosine on
    the like-matrix). A movie scores high if it is similar to movies the user already liked."""

    name = "Item-KNN"

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, pos: sp.csr_matrix):
        co = (pos.T @ pos).toarray().astype(np.float32)
        norm = np.sqrt(np.diag(co))
        sim = co / (np.outer(norm, norm) + self.shrink)  # shrinkage damps similarities built on few users
        np.fill_diagonal(sim, 0)
        keep = np.argpartition(-sim, self.k, axis=1)[:, : self.k]
        pruned = np.zeros_like(sim)
        rows = np.arange(sim.shape[0])[:, None]
        pruned[rows, keep] = sim[rows, keep]
        self.sim = pruned  # sim[j, h]: how strongly h supports recommending j
        self.pop = np.asarray(pos.sum(axis=0)).ravel()
        return self

    def score(self, hist: sp.csr_matrix, signed=None) -> np.ndarray:
        return np.asarray(hist @ self.sim.T)

    def contributions(self, hist_items: np.ndarray, item: int) -> np.ndarray:
        """Exact additive split of the score of `item` over the user's liked movies."""
        return self.sim[item, hist_items]


# ------------------------------------------------------------------ implicit-feedback ALS
class ALS:
    """Matrix factorisation for implicit feedback (Hu, Koren & Volinsky 2008), written in numpy.

    Every user and movie gets a `factors`-dimensional vector; a liked movie is a preference of 1 with
    confidence 1 + alpha, everything else a preference of 0 with confidence 1. Alternating least squares
    solves each side in closed form. The user vector is never stored: it is *folded in* from whatever
    history is given, so new users need no retraining."""

    name = "ALS matrix factorisation"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 20.0, iters: int = 12, seed: int = 42):
        self.factors, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed
        self.Y: np.ndarray | None = None

    # -- training
    def _solve_side(self, X: sp.csr_matrix, F: np.ndarray) -> np.ndarray:
        n, f = X.shape[0], F.shape[1]
        out = np.zeros((n, f), dtype=np.float32)
        G = F.T @ F + self.reg * np.eye(f, dtype=np.float32)
        for i in range(n):
            idx = X.indices[X.indptr[i]: X.indptr[i + 1]]
            if len(idx) == 0:
                continue
            Fs = F[idx]
            out[i] = np.linalg.solve(G + self.alpha * (Fs.T @ Fs), (1.0 + self.alpha) * Fs.sum(axis=0))
        return out

    def fit(self, pos: sp.csr_matrix):
        rng = np.random.default_rng(self.seed)
        self.Y = (rng.standard_normal((pos.shape[1], self.factors)) * 0.01).astype(np.float32)
        Xt = pos.T.tocsr()
        for _ in range(self.iters):
            U = self._solve_side(pos, self.Y)
            self.Y = self._solve_side(Xt, U)
        self._prepare()
        return self

    @classmethod
    def from_factors(cls, Y: np.ndarray, reg: float, alpha: float) -> ALS:
        m = cls(factors=Y.shape[1], reg=reg, alpha=alpha)
        m.Y = Y.astype(np.float32)
        m._prepare()
        return m

    def _prepare(self):
        self.G = self.Y.T @ self.Y + self.reg * np.eye(self.factors, dtype=np.float32)

    # -- inference
    def fold_in(self, hist: sp.csr_matrix) -> np.ndarray:
        """User vectors for a batch of histories (rows of a binary matrix)."""
        hist = hist.tocsr()
        out = np.zeros((hist.shape[0], self.factors), dtype=np.float32)
        for i in range(hist.shape[0]):
            idx = hist.indices[hist.indptr[i]: hist.indptr[i + 1]]
            if len(idx):
                Ys = self.Y[idx]
                out[i] = np.linalg.solve(self.G + self.alpha * (Ys.T @ Ys), (1.0 + self.alpha) * Ys.sum(axis=0))
        return out

    def score(self, hist: sp.csr_matrix, signed=None) -> np.ndarray:
        return self.fold_in(hist) @ self.Y.T

    def contributions(self, hist_items: np.ndarray, item: int) -> np.ndarray:
        """Exact additive split of the score: score(item) = sum over liked movies h of
        (1 + alpha) * y_h^T A^-1 y_item, where A is the user's normal-equation matrix. So each liked movie
        gets a signed share of the recommendation."""
        Ys = self.Y[hist_items]
        A = self.G + self.alpha * (Ys.T @ Ys)
        return (1.0 + self.alpha) * (Ys @ np.linalg.solve(A, self.Y[item]))


# ------------------------------------------------------------------ content based
class Content:
    """Recommend movies whose *attributes* resemble what the user liked: genres, era and title words
    (franchise signal). Needs no interaction data about the candidate movie, so it works for brand-new
    movies, the case where collaborative filtering has nothing to go on."""

    name = "Content-based"

    def __init__(self, movies, w_genre: float = 1.0, w_year: float = 0.5, w_title: float = 0.5):
        self.movies, self.w = movies, (w_genre, w_year, w_title)
        self.F = self._features()

    def _features(self) -> sp.csr_matrix:
        m = self.movies
        genres = data.genre_list(m)
        gi = {g: i for i, g in enumerate(genres)}
        rows = [i for i, gs in enumerate(m.genres) for _ in gs]
        cols = [gi[g] for gs in m.genres for g in gs]
        G = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(m), len(genres)))
        G = _unit(G)
        decade = ((m.year.fillna(1990) // 10) * 10).astype(int)
        D = sp.csr_matrix(
            (np.ones(len(m), dtype=np.float32), (np.arange(len(m)), pd_codes(decade))),
            shape=(len(m), decade.nunique()))
        tf = TfidfVectorizer(min_df=2, token_pattern=r"[a-z0-9']+", dtype=np.float32)
        T = tf.fit_transform(m.title.map(data.title_tokens))
        blocks = [b * w for b, w in zip((G, D, T), self.w, strict=True) if w > 0]
        return _unit(sp.hstack(blocks).tocsr())

    def score(self, hist: sp.csr_matrix, signed=None) -> np.ndarray:
        """`signed` (ratings - 3) lets disliked movies push similar ones down; without it, likes only."""
        w = signed if signed is not None else hist
        profile = _unit(sp.csr_matrix(w @ self.F))
        return np.asarray((profile @ self.F.T).todense())

    def contributions(self, hist_items: np.ndarray, item: int) -> np.ndarray:
        """How similar each liked movie is (cosine on the attribute vectors) to the recommended one."""
        return np.asarray((self.F[hist_items] @ self.F[item].T).todense()).ravel()

    def similar(self, item: int, k: int = 10) -> np.ndarray:
        s = np.asarray((self.F[item] @ self.F.T).todense()).ravel()
        s[item] = -1
        return np.argsort(-s)[:k]


def pd_codes(series) -> np.ndarray:
    return np.unique(series.to_numpy(), return_inverse=True)[1]


def _unit(m: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ m


# ------------------------------------------------------------------ hybrid
class Hybrid:
    """Weighted blend of per-user standardised scores from several models. Standardising first puts models
    with very different score scales (dot products, similarity sums, cosines) on one footing."""

    name = "Hybrid (ALS + item-KNN)"

    def __init__(self, parts: list[tuple[object, float]]):
        self.parts = parts

    def score(self, hist: sp.csr_matrix, signed=None) -> np.ndarray:
        return sum(w * zscore_rows(m.score(hist, signed)) for m, w in self.parts)
