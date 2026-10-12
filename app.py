"""Streamlit demo UI for the thesis recommender.

Run:  streamlit run app.py
First start trains the models (a few minutes) and saves them to models/;
later starts load the saved files instantly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_all  # noqa: E402
from evaluator import make_splits  # noqa: E402
from hybrid_engine import (  # noqa: E402
    ContentEncoder,
    HybridModel,
    ItemKNNModel,
    PopularityModel,
    SVDModel,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ALPHA = 0.7  # from the tuning run (logs/grid_search.json)


@st.cache_resource
def load_data():
    ratings, metadata = load_all(ROOT / "data" / "ratings.csv",
                                 ROOT / "data" / "metadata.csv")
    train, val, test, _ = make_splits(ratings)
    titles = dict(zip(metadata["item_id"].astype(str),
                      metadata["title"].astype(str)))
    return ratings, train, val, test, titles


@st.cache_resource
def load_models(_train, n_users, n_items):
    pop_path = MODEL_DIR / "popularity.npz"
    if pop_path.is_file():
        pop = PopularityModel.load(pop_path)
        knn = ItemKNNModel.load(MODEL_DIR / "itemknn.npz", _train)
        svd = SVDModel.load(MODEL_DIR / "svd.npz")
        content = ContentEncoder.load(MODEL_DIR / "content.pt")
    else:
        with st.spinner("First run: training models (takes a few minutes, only once)..."):
            sys.path.insert(0, str(ROOT / "scripts"))
            import train_models
            train_and_save = train_models.train_and_save
            train_and_save()
        pop = PopularityModel.load(pop_path)
        knn = ItemKNNModel.load(MODEL_DIR / "itemknn.npz", _train)
        svd = SVDModel.load(MODEL_DIR / "svd.npz")
        content = ContentEncoder.load(MODEL_DIR / "content.pt")
    return {
        "popularity": pop,
        "itemknn": knn,
        "svd": svd,
        "content": content,
        "hybrid": HybridModel(svd, content, alpha=ALPHA),
    }


def top_n_for_user(model, u_idx, train_items, val_items, titles, item_ids, n):
    scores = model.score_all_items(u_idx).astype(float)
    excluded = train_items.get(u_idx, set()) | val_items.get(u_idx, set())
    if excluded:
        scores[list(excluded)] = -np.inf
    order = np.argsort(-scores, kind="stable")[:n]
    rows = []
    for rank, i in enumerate(order, 1):
        iid = str(item_ids[i])
        rows.append({"#": rank,
                     "title": titles.get(iid, iid)[:90],
                     "score": round(float(scores[i]), 4)})
    return pd.DataFrame(rows)


st.set_page_config(page_title="Hybrid Recommender — thesis demo", layout="wide")
st.title("Hybrid Recommender — thesis demo")

ratings, train, val, test, titles = load_data()
n_users = int(ratings["user_id"].nunique())
n_items = int(ratings["item_id"].nunique())
models = load_models(train, n_users, n_items)

user_ids = np.sort(ratings["user_id"].unique())
item_ids = np.sort(ratings["item_id"].unique())
user_to_idx = {u: k for k, u in enumerate(user_ids)}
train_items = {int(u): set(g["i"].astype(int)) for u, g in train.groupby("u")}
val_items = {int(u): set(g["i"].astype(int)) for u, g in val.groupby("u")}

tab_rec, tab_results, tab_about = st.tabs(["Recommend", "Results", "About"])

with tab_rec:
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("random user"):
            st.session_state["uid"] = str(np.random.choice(user_ids))
        uid = st.text_input("user id", value=st.session_state.get("uid", str(user_ids[0])))
        model_name = st.selectbox("model", list(models.keys()), index=4)
        n = st.slider("how many", 5, 20, 10)
    with col2:
        if uid not in user_to_idx:
            st.error("unknown user id")
        else:
            u = user_to_idx[uid]
            liked = train[(train["u"] == u) & (train["rating"] >= 4.0)]
            st.subheader("items this user liked (train)")
            if len(liked):
                st.write("; ".join(
                    titles.get(str(item_ids[i]), str(item_ids[i]))[:60]
                    for i in liked["i"].head(8)))
            else:
                st.write("none rated 4+ in training")
            st.subheader(f"top {n} — {model_name}")
            df = top_n_for_user(models[model_name], u, train_items, val_items,
                                titles, item_ids, n)
            st.dataframe(df, use_container_width=True)

with tab_results:
    st.subheader("test results (mean ± 95% CI, 5 seeds)")
    import json
    summary_path = ROOT / "logs" / "eval_summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text())
        rows = []
        for name, metrics in payload["summary"].items():
            row = {"model": name}
            for m, agg in metrics.items():
                row[m] = f"{agg['mean']:.4f} ± {agg['ci95']:.4f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("no logs/eval_summary.json yet — run src/evaluator.py first")

with tab_about:
    st.write(
        "Weighted hybrid of collaborative filtering (SVD) and dense neural "
        "content embeddings, evaluated on Amazon Electronics. Hybrid score = "
        f"α·norm(CF) + (1−α)·norm(content), α={ALPHA}, picked by grid search "
        "on the validation split. Full methodology: REPORT.md / thesis/."
    )
