"""'Why was this recommended?' For the collaborative models the answer is exact, not a story:
the score of a recommended movie splits additively over the movies the user liked."""

from __future__ import annotations

from collections import Counter

import numpy as np


def because_you_liked(model, hist_items: np.ndarray, item: int, k: int = 3) -> list[tuple[int, float]]:
    """The liked movies that contributed most to recommending `item`, with their share of the total
    positive contribution. `model` is an ALS or ItemKNN instance (both expose `contributions`)."""
    hist_items = np.asarray(hist_items)
    if len(hist_items) == 0:
        return []
    c = model.contributions(hist_items, item)
    pos = np.clip(c, 0, None)
    total = pos.sum()
    if total <= 0:
        return []
    order = np.argsort(-pos)[:k]
    return [(int(hist_items[i]), float(pos[i] / total)) for i in order if pos[i] > 0]


def shared_genres(movies, hist_items: np.ndarray, item: int) -> list[tuple[str, int]]:
    """Genres of the recommended movie, with how many of the user's liked movies share each one."""
    counts = Counter(g for i in hist_items for g in movies.genres.iloc[int(i)])
    return sorted(((g, counts.get(g, 0)) for g in movies.genres.iloc[item]), key=lambda t: -t[1])
