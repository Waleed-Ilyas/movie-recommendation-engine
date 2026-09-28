"""Tune on the validation slice, refit on train+validation, grade once on the held-out future.

    python -m recsys.train          # ~15 min on a laptop CPU; writes artifacts/ and logs to MLflow
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import mlflow
import numpy as np

from . import cold, config, metrics
from . import experiment as ex
from . import models as M

CI = ("ndcg", "recall", "precision", "hit")


def _mean(per):
    return {k: float(v.mean()) for k, v in per.items()}


def tune(S: ex.Splits, log) -> dict:
    """Every choice here is made on the validation slice (last 10% of each user's timeline)."""
    users = ex.eval_users(S.pos_val)
    pop = np.asarray(S.pos_train.sum(0)).ravel()
    trials, best = [], {}

    def grade(model, name, params):
        t = time.time()
        per, cat, _ = ex.evaluate(model, S.pos_train, S.signed_train, S.seen_train, S.pos_val, pop, users)
        row = {"model": name, "params": params, **_mean(per), **cat, "seconds": round(time.time() - t, 1)}
        trials.append(row)
        log(name, params, round(row["ndcg"], 4))
        with mlflow.start_run(run_name=f"tune-{name}", nested=True):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({"val_ndcg10": row["ndcg"], "val_recall10": row["recall"]})
        return row

    grade(M.Popularity().fit(S.pos_train), "popularity", {})

    # Item-KNN
    for k, sh in itertools.product((20, 50, 100, 200), (0.0, 10.0, 50.0)):
        grade(M.ItemKNN(k, sh).fit(S.pos_train), "item_knn", {"k": k, "shrink": sh})
    best["item_knn"] = max((t for t in trials if t["model"] == "item_knn"), key=lambda t: t["ndcg"])["params"]

    # ALS coordinate search: confidence alpha first, then factors, then the regulariser
    def als_trial(f, a, r):
        return grade(M.ALS(f, r, a, iters=10).fit(S.pos_train), "als", {"factors": f, "alpha": a, "reg": r})

    def best_als():
        return max((t for t in trials if t["model"] == "als"), key=lambda t: t["ndcg"])["params"]

    for a in (1.0, 3.0, 10.0, 40.0):
        als_trial(64, a, 0.1)
    for f in (16, 32):
        als_trial(f, best_als()["alpha"], 0.1)
    for r in (0.01, 0.3, 1.0):
        als_trial(best_als()["factors"], best_als()["alpha"], r)
    best["als"] = max((t for t in trials if t["model"] == "als"), key=lambda t: t["ndcg"])["params"]

    # Content: which attribute blocks help?
    content_cfgs = {"genres": (1, 0, 0), "genres+era": (1, 0.5, 0), "genres+title": (1, 0, 0.5),
                    "genres+era+title": (1, 0.5, 0.5)}
    for name, w in content_cfgs.items():
        grade(M.Content(S.ds.movies, *w), "content", {"blocks": name, "w": w})
    best["content"] = max((t for t in trials if t["model"] == "content"), key=lambda t: t["ndcg"])["params"]

    # Hybrid: blend weights over standardised scores (cached once, blended cheaply)
    knn = M.ItemKNN(**best["item_knn"]).fit(S.pos_train)
    als = M.ALS(best["als"]["factors"], best["als"]["reg"], best["als"]["alpha"], iters=10).fit(S.pos_train)
    con = M.Content(S.ds.movies, *best["content"]["w"])
    h, sg = S.pos_train[users], S.signed_train[users]
    z = {n: M.zscore_rows(m.score(h, sg)) for n, m in (("als", als), ("knn", knn), ("con", con))}
    excl = S.seen_train[users]
    rel = S.pos_val[users]
    hyb = []
    for wk, wc in itertools.product((0.0, 0.5, 1.0), (0.0, 0.1, 0.25, 0.5)):
        s = z["als"] + wk * z["knn"] + wc * z["con"]
        per = metrics.per_user(metrics.top_k(s, excl), rel)
        row = {"model": "hybrid", "params": {"w_knn": wk, "w_content": wc}, **_mean(per)}
        hyb.append(row)
        log("hybrid", row["params"], round(row["ndcg"], 4))
    trials += hyb
    best["hybrid"] = max(hyb, key=lambda t: t["ndcg"])["params"]
    return {"trials": trials, "best": best}


def build_models(S: ex.Splits, best: dict, hist: str = "trainval"):
    """Fit every model on the given history slice with its tuned settings."""
    pos = S.pos_trainval if hist == "trainval" else S.pos_train
    pop = M.Popularity().fit(pos)
    knn = M.ItemKNN(**best["item_knn"]).fit(pos)
    als = M.ALS(best["als"]["factors"], best["als"]["reg"], best["als"]["alpha"], iters=10).fit(pos)
    con = M.Content(S.ds.movies, *best["content"]["w"])
    hp = best["hybrid"]
    hyb = M.Hybrid([(als, 1.0), (knn, hp["w_knn"]), (con, hp["w_content"])])
    return {"popularity": pop, "item_knn": knn, "als": als, "content": con, "hybrid": hyb}


class RandomModel:
    def __init__(self, n_items, seed=config.SEED):
        self.rng, self.n = np.random.default_rng(seed), n_items

    def score(self, hist, signed=None):
        return self.rng.random((hist.shape[0], self.n))


def final_test(S: ex.Splits, models: dict, log) -> dict:
    users = ex.eval_users(S.pos_test)
    pop = np.asarray(S.pos_trainval.sum(0)).ravel()
    out, per_all = {}, {}
    for name, m in {"random": RandomModel(S.ds.n_items), **models}.items():
        per, cat, _ = ex.evaluate(m, S.pos_trainval, S.signed_trainval, S.seen_trainval, S.pos_test, pop, users)
        per_all[name] = per
        out[name] = {**_mean(per), **cat, **{f"{k}_ci": metrics.bootstrap_mean(per[k]) for k in CI}}
        log("test", name, round(out[name]["ndcg"], 4))
    hist_len = np.diff(S.seen_trainval.indptr)[users]
    bands = {"light (<40 ratings)": hist_len < 40, "medium (40-120)": (hist_len >= 40) & (hist_len <= 120),
             "heavy (>120)": hist_len > 120}
    segments = {b: {"n_users": int(mask.sum()),
                    **{n: float(per_all[n]["ndcg"][mask].mean()) for n in ("popularity", "item_knn", "als", "hybrid")}}
                for b, mask in bands.items()}
    diffs = {}
    for a, b in (("hybrid", "popularity"), ("als", "popularity"), ("item_knn", "popularity"),
                 ("hybrid", "als"), ("als", "item_knn"), ("hybrid", "item_knn")):
        d = per_all[a]["ndcg"] - per_all[b]["ndcg"]
        diffs[f"{a}_minus_{b}"] = {"mean": float(d.mean()), "ci": metrics.bootstrap_mean(d)}
    return {"models": out, "n_users": int(len(users)), "segments": segments, "paired_ndcg_diffs": diffs}


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("recsys")

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    S = ex.build()
    log("data", S.ds.n_users, "users", S.ds.n_items, "items", len(S.ds.ratings), "ratings")
    with mlflow.start_run(run_name="pipeline"):
        cache = config.ARTIFACTS / "tuning.json"
        if cache.exists() and "--retune" not in sys.argv:
            tuned = json.loads(cache.read_text(encoding="utf-8"))  # validation search is the slow part
            log("loaded cached tuning")
        else:
            tuned = tune(S, log)
            cache.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
        best = tuned["best"]
        log("best", json.dumps(best))
        models = build_models(S, best)
        test = final_test(S, models, log)
        log("cold start: new users")
        blend = cold.tune_pop_blend(S, build_models(S, best, hist="train"), log)
        new_user = cold.new_user_sweep(S, models, blend, log)
        log("cold start: new items")
        new_item = cold.new_item_eval(S, best, log)
        results = {
            "dataset": {"users": S.ds.n_users, "items": S.ds.n_items, "ratings": int(len(S.ds.ratings)),
                        "positive_rate": float((S.ds.ratings.rating >= config.POSITIVE).mean()),
                        "split": config.SPLIT, "k": config.K, "positive_threshold": config.POSITIVE},
            "tuning": tuned, "test": test, "cold_start_users": new_user, "cold_start_blend_weights": blend, "cold_start_items": new_item,
        }
        (config.ARTIFACTS / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        np.save(config.ARTIFACTS / "item_factors.npy", models["als"].Y.astype(np.float32))
        (config.ARTIFACTS / "hyperparams.json").write_text(
            json.dumps({**best, "cold_blend": blend}, indent=1), encoding="utf-8")
        for name, r in test["models"].items():
            mlflow.log_metrics({f"test_{name}_ndcg10": r["ndcg"], f"test_{name}_recall10": r["recall"]})
    log("done")


if __name__ == "__main__":
    main()
