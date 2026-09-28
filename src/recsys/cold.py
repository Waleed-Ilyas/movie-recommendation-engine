"""Cold-start experiments: what happens with almost no information?

New user: the model knows only the first m movies a user liked (an onboarding "pick 5 favourites" screen).
New item: movies with zero ratings in training, which collaborative filtering cannot score at all."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config, data, metrics
from . import experiment as ex
from . import models as M

USER_M = (0, 1, 3, 5, 10, 20, 50)
COLD_ITEM_FRACTION = 0.10


def first_m_positives(S: ex.Splits, users: np.ndarray, m: int, slices=("train", "val")) -> sp.csr_matrix:
    """Each user's m earliest liked movies from the given slices, as a history matrix."""
    r = S.ds.ratings
    if m == 0:
        return sp.csr_matrix((len(users), S.ds.n_items), dtype=np.float32)
    lab = data.chronological_split(r)
    r = r[lab.isin(slices) & (r.rating >= config.POSITIVE) & r.user.isin(users)]
    r = r.sort_values(["user", "ts"], kind="stable")
    first = r[r.groupby("user").cumcount() < m]
    row = np.searchsorted(users, first.user.to_numpy())
    return sp.csr_matrix((np.ones(len(first), dtype=np.float32), (row, first.item.to_numpy())),
                         shape=(len(users), S.ds.n_items))


BLEND_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)


class _Blend:
    """Standardised model score plus a popularity prior: lean on what everyone likes until the user has
    told us enough about themselves."""

    def __init__(self, model, pop, w):
        self.model, self.pop, self.w = model, pop, w

    def score(self, hist, signed=None):
        return M.zscore_rows(self.model.score(hist, signed)) + self.w * M.zscore_rows(self.pop.score(hist, signed))


def tune_pop_blend(S: ex.Splits, models_train: dict, log) -> dict:
    """For each history length m, the popularity-prior weight with the best NDCG@10 on the validation slice."""
    users = ex.eval_users(S.pos_val)
    exclude, rel = S.seen_train[users], S.pos_val[users]
    out = {}
    for m in USER_M[1:]:
        hist = first_m_positives(S, users, m, slices=("train",))
        scores = {}
        for w in BLEND_GRID:
            b = _Blend(models_train["hybrid"], models_train["popularity"], w)
            scores[w] = float(metrics.per_user(metrics.top_k(b.score(hist, hist), exclude), rel)["ndcg"].mean())
        out[str(m)] = max(scores, key=scores.get)
        log("blend weight m=", m, out[str(m)], {k: round(v, 4) for k, v in scores.items()})
    return out


def new_user_sweep(S: ex.Splits, models: dict, blend: dict, log) -> dict:
    """NDCG@10 on the held-out future when the model only knows the user's first m liked movies.
    m = 0 is popularity for everyone (the only thing a model can do with no information).
    `hybrid_pop` adds the validation-tuned popularity prior."""
    users = ex.eval_users(S.pos_test)
    exclude, rel = S.seen_trainval[users], S.pos_test[users]
    out = {}
    for m in USER_M:
        hist = first_m_positives(S, users, m)
        row = {}
        candidates = {name: models[name] for name in ("popularity", "item_knn", "als", "content", "hybrid")}
        candidates["hybrid_pop"] = _Blend(models["hybrid"], models["popularity"], blend.get(str(m), 0.0))
        for name, model in candidates.items():
            model = models["popularity"] if m == 0 else model
            per = metrics.per_user(metrics.top_k(model.score(hist, hist), exclude), rel)
            row[name] = {"ndcg": float(per["ndcg"].mean()), "recall": float(per["recall"].mean()),
                         "ndcg_ci": metrics.bootstrap_mean(per["ndcg"], n=300)}
        out[str(m)] = row
        log("new user m=", m, {k: round(v["ndcg"], 4) for k, v in row.items()})
    return out


def new_item_eval(S: ex.Splits, best: dict, log) -> dict:
    """Hide a random 10% of the catalogue from training entirely, then rank *only those movies* for each
    user by content similarity to what they liked. Collaborative models have no signal for them."""
    rng = np.random.default_rng(config.SEED)
    n = S.ds.n_items
    cold = np.sort(rng.choice(n, int(n * COLD_ITEM_FRACTION), replace=False))
    keep = np.ones(n, dtype=bool)
    keep[cold] = False
    mask = sp.diags(keep.astype(np.float32))
    hist = (S.pos_trainval @ mask).tocsr()
    signed = (S.signed_trainval @ mask).tocsr()
    rel_full = (S.pos_test @ sp.diags((~keep).astype(np.float32))).tocsr()  # test likes of cold movies only
    users = ex.eval_users(rel_full)
    # candidates are the cold movies only: everything else is excluded, as is anything the user already rated
    not_cold = sp.csr_matrix(np.tile(keep, (len(users), 1)).astype(np.float32))
    exclude = (S.seen_trainval[users] + not_cold).tocsr()
    exclude.data[:] = 1
    rel = rel_full[users]
    out = {"n_cold_items": int(len(cold)), "n_users": int(len(users)),
           "avg_cold_liked_per_user": float(np.diff(rel.indptr).mean()), "variants": {}}
    rng2 = np.random.default_rng(1)
    variants = {"random": None, "genres": (1, 0, 0), "genres+era": (1, 0.5, 0), "genres+title": (1, 0, 0.5),
                "genres+era+title": (1, 0.5, 0.5)}
    for name, w in variants.items():
        if w is None:
            scores = rng2.random((len(users), n))
        else:
            scores = M.Content(S.ds.movies, *w).score(hist[users], signed[users])
        per = metrics.per_user(metrics.top_k(scores, exclude), rel)
        out["variants"][name] = {k: float(v.mean()) for k, v in per.items()}
        out["variants"][name]["ndcg_ci"] = metrics.bootstrap_mean(per["ndcg"], n=300)
        log("new item", name, round(out["variants"][name]["ndcg"], 4))
    return out
