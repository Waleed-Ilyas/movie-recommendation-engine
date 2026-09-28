"""README figures from artifacts/results.json. Run: python -m recsys.report"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402

GOLD, CYAN, IVORY, BG, GREY, RED, VIOLET = "#D9B26A", "#4FD1C5", "#F2EDE4", "#0A0A0C", "#8a8578", "#E0625A", "#b08fe8"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": IVORY,
                     "axes.labelcolor": IVORY, "xtick.color": IVORY, "ytick.color": IVORY,
                     "axes.edgecolor": GREY, "font.size": 11})
NAMES = {"random": "Random", "popularity": "Popularity", "content": "Content-based", "item_knn": "Item-KNN",
         "als": "ALS", "hybrid": "Hybrid (ALS + item-KNN)", "hybrid_pop": "Hybrid + popularity prior"}
COLORS = {"random": GREY, "popularity": GREY, "content": RED, "item_knn": CYAN, "als": VIOLET, "hybrid": GOLD, "hybrid_pop": IVORY}


def fig_compare(res) -> None:
    t = res["test"]["models"]
    ks = [k for k in NAMES if k in t]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(ks))[::-1]
    for yi, k in zip(y, ks, strict=True):
        v = t[k]["ndcg"]
        lo, hi = t[k]["ndcg_ci"]
        ax.barh(yi, v, color=COLORS[k], height=0.6)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=IVORY, capsize=3, lw=1)
        ax.text(hi + 0.002, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, [NAMES[k] for k in ks])
    ax.set_xlabel("NDCG@10 on the held-out future, 95% interval over users")
    ax.set_title(f"Ranking quality, {res['test']['n_users']:,} users", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "01_model_comparison.png", dpi=150)


def fig_cold_users(res) -> None:
    cs = res["cold_start_users"]
    ms = sorted(int(m) for m in cs)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for k in ("popularity", "content", "item_knn", "als", "hybrid", "hybrid_pop"):
        ax.plot(ms, [cs[str(m)][k]["ndcg"] for m in ms], marker="o", color=COLORS[k],
                lw=2.6 if k == "hybrid" else 1.5, label=NAMES[k])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.set_xlabel("liked movies the model knows about the user (0 = brand-new user)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("New-user cold start: how fast does personalisation pay off?", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "02_cold_start_users.png", dpi=150)


def fig_segments(res) -> None:
    seg = res["test"]["segments"]
    bands = list(seg)
    ks = ("popularity", "item_knn", "als", "hybrid")
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for j, k in enumerate(ks):
        ax.bar(np.arange(len(bands)) + (j - 1.5) * w, [seg[b][k] for b in bands], w, color=COLORS[k], label=NAMES[k])
    ax.set_xticks(range(len(bands)), [f"{b}\n(n={seg[b]['n_users']:,})" for b in bands])
    ax.set_ylabel("NDCG@10")
    ax.set_title("Who benefits from personalisation? Quality by how much history the user has", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "03_segments.png", dpi=150)


def fig_tradeoff(res) -> None:
    t = res["test"]["models"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k in ("popularity", "content", "item_knn", "als", "hybrid"):
        ax.scatter(t[k]["coverage"] * 100, t[k]["ndcg"], s=140, color=COLORS[k], zorder=3)
        ax.annotate(NAMES[k], (t[k]["coverage"] * 100, t[k]["ndcg"]), textcoords="offset points", xytext=(8, 6),
                    fontsize=9)
    ax.set_xlabel("catalogue coverage: % of all movies ever recommended (higher = less same-list-for-everyone)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. variety", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "04_accuracy_vs_coverage.png", dpi=150)


if __name__ == "__main__":
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    r = json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    fig_compare(r)
    fig_cold_users(r)
    fig_segments(r)
    fig_tradeoff(r)
    print("figures written")
