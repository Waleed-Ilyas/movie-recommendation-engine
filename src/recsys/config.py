"""Paths and experiment constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

SEED = 42
K = 10  # recommendations shown / evaluated
POSITIVE = 4  # rating >= 4 counts as "liked"
SPLIT = (0.7, 0.1, 0.2)  # per-user chronological: train / validation / test
MIN_EVAL_POSITIVES = 1  # a user needs >= 1 liked movie in the evaluated slice
