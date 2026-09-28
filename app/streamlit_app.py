"""Movie recommender demo: real MovieLens users with 'did they later like it?' checks, a build-your-own-taste
cold-start mode, and the evaluation results. MovieLens is downloaded on first run (its licence forbids
redistribution); only the trained item factors are committed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys import config, metrics  # noqa: E402
from recsys import experiment as ex  # noqa: E402
from recsys import models as M  # noqa: E402
from recsys.explain import because_you_liked, shared_genres  # noqa: E402

st.set_page_config(page_title="Movie recommender", page_icon="🎬", layout="wide")
GOLD, CYAN, RED, GREY = "#D9B26A", "#4FD1C5", "#E0625A", "#8a8578"
MODEL_LABELS = {"hybrid": "Hybrid (ALS + item-KNN)", "als": "ALS matrix factorisation",
                "item_knn": "Item-KNN (collaborative)", "content": "Content-based (genre, era, title)",
                "popularity": "Popularity (same list for everyone)"}
PRESETS = {
    "Sci-fi fan": ["Star Wars: Episode IV - A New Hope", "Alien", "Blade Runner", "Terminator 2: Judgment Day",
                   "Matrix, The"],
    "Family & animation": ["Toy Story", "Lion King, The", "Aladdin", "Beauty and the Beast", "Toy Story 2"],
    "Crime & drama": ["Godfather, The", "Pulp Fiction", "Goodfellas", "Usual Suspects, The",
                      "Shawshank Redemption, The"],
    "Romantic comedy": ["Princess Bride, The", "When Harry Met Sally...", "Groundhog Day", "Clueless",
                        "Sleepless in Seattle"],
}


@st.cache_resource(show_spinner="First run: downloading MovieLens 1M and building the models (~30 s)...")
def load():
    S = ex.build()
    res = json.loads((config.ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    best = json.loads((config.ARTIFACTS / "hyperparams.json").read_text(encoding="utf-8"))
    pos = S.pos_trainval
    knn = M.ItemKNN(**best["item_knn"]).fit(pos)
    als = M.ALS.from_factors(np.load(config.ARTIFACTS / "item_factors.npy"), best["als"]["reg"],
                             best["als"]["alpha"])
    con = M.Content(S.ds.movies, *best["content"]["w"])
    hp = best["hybrid"]
    models = {"popularity": M.Popularity().fit(pos), "item_knn": knn, "als": als, "content": con,
              "hybrid": M.Hybrid([(als, 1.0), (knn, hp["w_knn"]), (con, hp["w_content"])])}
    blend = {int(m): w for m, w in best["cold_blend"].items()}
    return S, res, models, blend


S, res, models, BLEND = load()
movies = S.ds.movies
label = [f"{t} ({int(y)})" if y == y else t for t, y in zip(movies.title, movies.year, strict=True)]
seen_label: dict[str, int] = {}
for i, lb in enumerate(label):
    if lb in seen_label:
        label[i] = f"{lb} #{movies.movie_id.iloc[i]}"
    seen_label[label[i]] = i
pop = np.asarray(S.pos_trainval.sum(0)).ravel()
test_users = ex.eval_users(S.pos_test)

with st.sidebar:
    st.header("🎬 Movie recommender")
    st.caption("MovieLens 1M: 1,000,209 ratings by 6,040 users of 3,706 movies. Each user's earliest 80% of "
               "ratings are what the model may see; the latest 20% are the future it is graded on.")
    model_key = st.selectbox("Recommender", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
    n_recs = st.slider("Recommendations", 5, 20, 10)
    st.divider()
    with st.expander("How to read this"):
        st.markdown(
            "- **Because you liked** is exact for the collaborative models: the score of a recommendation is "
            "split additively over the movies you liked, and the biggest shares are shown.\n"
            "- **✔ later liked** (real-user tab) means the user really did rate that movie 4+ stars *after* the "
            "cut-off. It is the honest check, but it under-counts: a good recommendation the user simply "
            "never got round to rating looks like a miss.\n"
            "- A rating of **4 or 5 stars counts as 'liked'**.")


def recommend(hist_items: np.ndarray, exclude_items: np.ndarray, key: str, k: int, cold_prior: bool = False):
    """Top-k for one history. cold_prior blends in popularity with the weight tuned for this history length."""
    h = sp.csr_matrix((np.ones(len(hist_items), dtype=np.float32), (np.zeros(len(hist_items), dtype=int), hist_items)),
                      shape=(1, S.ds.n_items))
    if len(hist_items) == 0:
        s = models["popularity"].score(h)
    elif cold_prior and key not in ("popularity", "content"):
        w = BLEND[max(m for m in BLEND if m <= len(hist_items))] if len(hist_items) >= min(BLEND) else BLEND[min(BLEND)]
        s = M.zscore_rows(models[key].score(h, h)) + w * M.zscore_rows(models["popularity"].score(h))
    else:
        s = models[key].score(h, h)
    ex_m = np.zeros((1, S.ds.n_items), dtype=np.float32)
    ex_m[0, exclude_items] = 1
    return metrics.top_k(s, sp.csr_matrix(ex_m), k)[0]


def explain_text(item: int, hist_items: np.ndarray, key: str) -> str:
    if key == "popularity" or len(hist_items) == 0:
        return f"One of the most-liked movies overall ({int(pop[item]):,} likes)"
    src = {"content": models["content"], "item_knn": models["item_knn"]}.get(key, models["als"])
    top = because_you_liked(src, hist_items, item, k=2)
    parts = [f"{movies.title.iloc[i]} ({s:.0%})" for i, s in top]
    shared = [f"{g} x{c}" for g, c in shared_genres(movies, hist_items, item) if c][:2]
    txt = "Because you liked " + ", ".join(parts) if parts else "Fits the overall pattern of your taste"
    return txt + (f" · shares {', '.join(shared)}" if shared else "")


def rec_table(items, hist_items, key, later=None) -> pd.DataFrame:
    rows = []
    for r, i in enumerate(items, 1):
        row = {"#": r, "Movie": f"{movies.title.iloc[i]} ({int(movies.year.iloc[i]) if movies.year.iloc[i] == movies.year.iloc[i] else '?'})",
               "Why": explain_text(int(i), hist_items, key)}
        if later is not None:
            row["Later liked?"] = "✔ yes" if i in later else ""
        rows.append(row)
    return pd.DataFrame(rows)


REC_COLUMNS = {"#": st.column_config.NumberColumn(width="small"), "Movie": st.column_config.TextColumn(width="medium"), "Why": st.column_config.TextColumn(width="large")}


st.title("Movie recommendation engine")
st.caption("Popularity, item-based collaborative filtering, implicit-feedback ALS, content-based and a hybrid, "
           "compared honestly on a held-out future.")
test = res["test"]["models"]
c = st.columns(4)
c[0].metric("Popularity NDCG@10", f"{test['popularity']['ndcg']:.3f}")
best_key = max(("item_knn", "als", "hybrid"), key=lambda k: test[k]["ndcg"])
c[1].metric(f"Best model ({MODEL_LABELS[best_key].split(' (')[0]})", f"{test[best_key]['ndcg']:.3f}",
            f"{100 * (test[best_key]['ndcg'] / test['popularity']['ndcg'] - 1):+.0f}% vs popularity",
            delta_arrow="off")
c[2].metric("Catalogue coverage: popularity → best", f"{test['popularity']['coverage']:.0%} → {test[best_key]['coverage']:.0%}")
c[3].metric("Users evaluated", f"{res['test']['n_users']:,}")

tab_user, tab_you, tab_eval = st.tabs(["A real user", "Build your own taste", "How well does it work?"])

# ------------------------------------------------------------------ real user
with tab_user:
    left, right = st.columns([1, 3])
    user_ids = [int(u) + 1 for u in test_users[:400]]
    uid = left.selectbox("MovieLens user", user_ids, key="uid", help="The first 400 of the evaluated users.")
    left.button("🎲 Random user", on_click=lambda: st.session_state.update(
        uid=int(np.random.default_rng().choice(user_ids))))
    u = uid - 1
    hist = S.seen_trainval[u].indices
    liked = S.pos_trainval[u].indices
    future = set(S.pos_test[u].indices)
    ratings_u = S.ds.ratings[S.ds.ratings.user == u].sort_values("ts")
    known = ratings_u[ratings_u.item.isin(hist)]  # what the model was allowed to see
    left.metric("Movies the model has seen them rate", len(hist))
    left.metric("Liked (4-5★)", len(liked))
    left.metric("Liked later (the future)", len(future))
    top_g = pd.Series([g for i in liked for g in movies.genres.iloc[i]]).value_counts().head(6)
    with right:
        st.markdown("**Recent things they liked before the cut-off**")
        recent = known[known.rating >= 4].tail(8).iloc[::-1]
        st.dataframe(pd.DataFrame({"Movie": [label[i] for i in recent.item], "Rating": [f"{r}★" for r in recent.rating],
                                   "Genres": [", ".join(movies.genres.iloc[i]) for i in recent.item]}),
                     hide_index=True, use_container_width=True)
        st.caption("Favourite genres: " + ", ".join(f"{g} ({n})" for g, n in top_g.items()))
    recs = recommend(liked, hist, model_key, n_recs)
    base = recommend(liked, hist, "popularity", n_recs)
    hits, base_hits = sum(int(i) in future for i in recs), sum(int(i) in future for i in base)
    st.subheader(f"Top {n_recs} from {MODEL_LABELS[model_key].split(' (')[0]}")
    st.markdown(f"**{hits} of {n_recs}** were movies this user really did like later "
                f"(popularity would have hit **{base_hits}**).")
    st.dataframe(rec_table(recs, liked, model_key, later=future), hide_index=True, use_container_width=True, column_config=REC_COLUMNS)

# ------------------------------------------------------------------ build your own
with tab_you:
    st.markdown("Pick a few movies you like and the recommender updates instantly. This is the **cold-start** case: "
                "a brand-new user with no history, the model just folds in your picks.")
    pcols = st.columns(len(PRESETS))
    for col, (name, titles) in zip(pcols, PRESETS.items(), strict=True):
        if col.button(name, use_container_width=True):
            by_title = {movies.title.iloc[i]: lb for i, lb in enumerate(label)}
            st.session_state["picks"] = [by_title[t] for t in titles if t in by_title]
    order = np.argsort(-pop)
    picks = st.multiselect("Movies you like", [label[i] for i in order], key="picks",
                           placeholder="Type to search 3,706 movies")
    idx = np.array([seen_label[p] for p in picks], dtype=int)
    if len(idx) == 0:
        st.info("Choose a preset or search for a few favourites. With no picks you just get the popular list.")
    use_prior = st.toggle("Blend in popularity while I have few picks", value=True,
                          help="The blend weight was tuned on validation data per number of known likes: the fewer "
                               "picks, the more it leans on what everyone likes.")
    recs = recommend(idx, idx, model_key, n_recs, cold_prior=use_prior)
    st.dataframe(rec_table(recs, idx, model_key), hide_index=True, use_container_width=True, column_config=REC_COLUMNS)
    m5 = res["cold_start_users"].get("5")
    if m5:
        st.caption(f"How good is this with few picks? On the held-out future, with only 5 liked movies known: "
                   f"hybrid alone NDCG@10 {m5['hybrid']['ndcg']:.3f}, popularity {m5['popularity']['ndcg']:.3f}, "
                   f"hybrid + popularity prior {m5['hybrid_pop']['ndcg']:.3f} (see the next tab).")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    order_keys = ["random", "popularity", "content", "item_knn", "als", "hybrid"]
    names = {"random": "Random", **{k: MODEL_LABELS[k].split(" (")[0] for k in MODEL_LABELS}}
    df = pd.DataFrame([{"Model": names[k], "NDCG@10": test[k]["ndcg"],
                        "95% interval": f"{test[k]['ndcg_ci'][0]:.3f}-{test[k]['ndcg_ci'][1]:.3f}",
                        "Recall@10": test[k]["recall"], "Precision@10": test[k]["precision"],
                        "Hit rate@10": test[k]["hit"], "Catalogue coverage": test[k]["coverage"],
                        "Novelty": test[k]["novelty"]} for k in order_keys])
    st.markdown(f"**Held-out future, {res['test']['n_users']:,} users, K = 10.** Tuned on a separate validation slice.")
    st.dataframe(df.style.format({"NDCG@10": "{:.4f}", "Recall@10": "{:.3f}", "Precision@10": "{:.3f}",
                                  "Hit rate@10": "{:.3f}", "Catalogue coverage": "{:.1%}", "Novelty": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    cs = res["cold_start_users"]
    ms = sorted(int(m) for m in cs)
    fig = go.Figure()
    names["hybrid_pop"] = "Hybrid + popularity prior"
    for k, col in (("popularity", GREY), ("content", RED), ("item_knn", CYAN), ("als", "#b08fe8"),
                   ("hybrid", GOLD), ("hybrid_pop", "#ffffff")):
        fig.add_trace(go.Scatter(x=ms, y=[cs[str(m)][k]["ndcg"] for m in ms], mode="lines+markers", name=names[k],
                                 line=dict(color=col, width=3 if k == "hybrid" else 1.6)))
    fig.update_layout(title="New user: NDCG@10 by number of liked movies known", xaxis_title="movies the model knows",
                      yaxis_title="NDCG@10", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#F2EDE4", margin=dict(l=0, r=0, t=40, b=0), xaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
    ni = res["cold_start_items"]
    st.markdown(f"**New movies with zero ratings** ({ni['n_cold_items']} hidden movies, {ni['n_users']:,} users): "
                "collaborative models cannot score them at all; content similarity can.")
    st.dataframe(pd.DataFrame([{"Ranker": k, "NDCG@10": v["ndcg"], "Recall@10": v["recall"]}
                               for k, v in ni["variants"].items()]).style.format({"NDCG@10": "{:.3f}", "Recall@10": "{:.3f}"}),
                 hide_index=True)
