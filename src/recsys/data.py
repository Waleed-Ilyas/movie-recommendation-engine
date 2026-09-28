"""MovieLens 1M: download, parse, per-user chronological split, sparse matrices.

The dataset licence forbids redistribution, so it is never committed: `download()` fetches it from
GroupLens on demand (the deployed app does the same on first run)."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config


def download() -> None:
    """Fetch and unzip ml-1m into data/raw/ml-1m (no-op if already there)."""
    target = config.DATA / "ml-1m"
    if (target / "ratings.dat").exists():
        return
    config.DATA.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.MOVIELENS_URL, timeout=120) as r:
        blob = r.read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall(config.DATA)


@dataclass
class Dataset:
    ratings: pd.DataFrame  # user, item (0-based contiguous), rating, ts
    movies: pd.DataFrame  # index = item, columns: movie_id, title, year, genres (list)
    n_users: int
    n_items: int


def _parse_movies(path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="::", engine="python", names=["movie_id", "raw", "genres"],
                    encoding="latin-1")
    year = m.raw.str.extract(r"\((\d{4})\)\s*$")[0]
    m["year"] = pd.to_numeric(year)
    m["title"] = m.raw.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    m["genres"] = m.genres.str.split("|")
    return m[["movie_id", "title", "year", "genres"]]


def load() -> Dataset:
    """Ratings with users and items re-indexed 0..n-1 (users by id, items by id order)."""
    download()
    base = config.DATA / "ml-1m"
    r = pd.read_csv(base / "ratings.dat", sep="::", engine="python",
                    names=["user_id", "movie_id", "rating", "ts"])
    movies = _parse_movies(base / "movies.dat")
    items = np.sort(r.movie_id.unique())  # only movies that were ever rated
    item_ix = pd.Series(np.arange(len(items)), index=items)
    users = np.sort(r.user_id.unique())
    user_ix = pd.Series(np.arange(len(users)), index=users)
    out = pd.DataFrame({"user": user_ix[r.user_id].to_numpy(), "item": item_ix[r.movie_id].to_numpy(),
                        "rating": r.rating.to_numpy(np.int8), "ts": r.ts.to_numpy(np.int64)})
    movies = movies.set_index("movie_id").loc[items].reset_index()
    movies.index.name = "item"
    return Dataset(out, movies, len(users), len(items))


def chronological_split(ratings: pd.DataFrame, fractions=config.SPLIT) -> pd.Series:
    """Label every rating 'train' / 'val' / 'test' by its position in that user's own timeline.

    Each user's earliest 70% of ratings train the model, the next 10% tune it, the last 20% are the
    held-out future. Ties on the timestamp keep the file order (a stable sort)."""
    order = ratings.sort_values(["user", "ts"], kind="stable")
    pos = order.groupby("user").cumcount()
    size = order.groupby("user")["item"].transform("size")
    frac = (pos + 0.5) / size
    a, b = fractions[0], fractions[0] + fractions[1]
    label = np.where(frac < a, "train", np.where(frac < b, "val", "test"))
    return pd.Series(label, index=order.index).reindex(ratings.index)


def to_csr(ratings: pd.DataFrame, n_users: int, n_items: int, *, positive_only: bool = True,
           threshold: int = config.POSITIVE) -> sp.csr_matrix:
    """Binary user x item matrix. positive_only=True keeps only 'liked' ratings; False marks any rating."""
    r = ratings[ratings.rating >= threshold] if positive_only else ratings
    m = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r.user.to_numpy(), r.item.to_numpy())),
                      shape=(n_users, n_items))
    m.sum_duplicates()
    return m


def genre_list(movies: pd.DataFrame) -> list[str]:
    return sorted({g for gs in movies.genres for g in gs})


_TOKEN = re.compile(r"[A-Za-z0-9']+")


def title_tokens(title: str) -> str:
    """Lower-cased title words for the franchise/sequel signal ('Star Trek II' ~ 'Star Trek III')."""
    return " ".join(t.lower() for t in _TOKEN.findall(title))
