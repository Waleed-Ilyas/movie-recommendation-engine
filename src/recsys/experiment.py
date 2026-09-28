"""Shared experiment plumbing: the split matrices and one function that scores a model on a slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import config, data, metrics


@dataclass
class Splits:
    ds: data.Dataset
    pos_train: sp.csr_matrix  # liked, train slice
    pos_trainval: sp.csr_matrix  # liked, train + validation slices (history at test time)
    pos_val: sp.csr_matrix
    pos_test: sp.csr_matrix
    seen_train: sp.csr_matrix  # any rating, train slice
    seen_trainval: sp.csr_matrix
    signed_train: sp.csr_matrix  # rating - 3, for the content model
    signed_trainval: sp.csr_matrix


def _signed(ratings, n_users, n_items) -> sp.csr_matrix:
    m = sp.csr_matrix(((ratings.rating.to_numpy() - 3).astype(np.float32),
                       (ratings.user.to_numpy(), ratings.item.to_numpy())), shape=(n_users, n_items))
    return m


def build(ds: data.Dataset | None = None) -> Splits:
    ds = ds or data.load()
    r = ds.ratings.assign(part=data.chronological_split(ds.ratings))
    nu, ni = ds.n_users, ds.n_items
    part = r.part
    tr, val, te = r[part == "train"], r[part == "val"], r[part == "test"]
    tv = r[part != "test"]
    return Splits(
        ds, data.to_csr(tr, nu, ni), data.to_csr(tv, nu, ni), data.to_csr(val, nu, ni), data.to_csr(te, nu, ni),
        data.to_csr(tr, nu, ni, positive_only=False), data.to_csr(tv, nu, ni, positive_only=False),
        _signed(tr, nu, ni), _signed(tv, nu, ni))


def eval_users(relevant: sp.csr_matrix) -> np.ndarray:
    """Users with at least one liked movie in the slice being predicted."""
    return np.flatnonzero(np.diff(relevant.indptr) >= config.MIN_EVAL_POSITIVES)


def evaluate(model, hist_pos, hist_signed, exclude, relevant, popularity, users=None, k=config.K):
    """Score `model` on the users' histories and grade the top-k against `relevant`.
    Returns (per-user metric arrays, catalogue stats, the recommendation matrix)."""
    users = eval_users(relevant) if users is None else users
    scores = model.score(hist_pos[users], hist_signed[users])
    recs = metrics.top_k(scores, exclude[users], k)
    per = metrics.per_user(recs, relevant[users], k)
    return per, metrics.catalogue_stats(recs, popularity, hist_pos.shape[1]), recs
