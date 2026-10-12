"""Storefront demo for the thesis recommender (Amazon-like).

Run:  streamlit run app.py
Needs data/ratings.csv + data/metadata.csv (with image_url, price columns
from scripts/add_product_media.py) and trained models in models/
(scripts/train_models.py). First start loads everything, later starts are
instant since models are saved to disk.
"""

from pathlib import Path

import html

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
CARDS_PER_ROW = 4

PLACEHOLDER_IMG = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E"
    "%3Crect width='300' height='300' fill='%23e8e8e8'/%3E"
    "%3Ctext x='150' y='155' font-family='sans-serif' font-size='20' "
    "fill='%23999' text-anchor='middle'%3Eno image%3C/text%3E%3C/svg%3E"
)


@st.cache_resource
def load_data():
    ratings, metadata = load_all(ROOT / "data" / "ratings.csv",
                                 ROOT / "data" / "metadata.csv")
    train, val, test, _ = make_splits(ratings)
    for col in ("image_url", "price"):
        if col not in metadata.columns:
            metadata[col] = ""
    info = {
        str(r["item_id"]): {
            "title": html.unescape(str(r["title"])).strip()[:120]
                     or str(r["item_id"]),
            "image": str(r["image_url"]) or PLACEHOLDER_IMG,
            "price": str(r["price"]),
            "desc": html.unescape(str(r["description"]))[:600],
        }
        for _, r in metadata.iterrows()
    }
    return ratings, train, val, test, info


@st.cache_resource
def load_models(_train):
    return {
        "popularity": PopularityModel.load(MODEL_DIR / "popularity.npz"),
        "itemknn": ItemKNNModel.load(MODEL_DIR / "itemknn.npz", _train),
        "svd": SVDModel.load(MODEL_DIR / "svd.npz"),
        "content": ContentEncoder.load(MODEL_DIR / "content.pt"),
    }


def get_hybrid(models):
    return HybridModel(models["svd"], models["content"], alpha=ALPHA)


def stars(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def price_str(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    return p if p.startswith("$") else f"${p}"


def card_html(item_id: str, info: dict, extra: str = "") -> str:
    title = info["title"].replace('"', "&quot;")
    img = info["image"].replace('"', "&quot;")
    price = price_str(info["price"])
    return f"""
    <div style="border:1px solid #e0e0e0;border-radius:8px;padding:10px;
                height:100%;background:white">
      <img src="{img}" onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
           style="width:100%;height:180px;object-fit:contain;background:#fafafa"/>
      <div style="font-weight:600;font-size:14px;margin:8px 0 4px;height:60px;
                  overflow:hidden">{title}</div>
      <div style="color:#B12704;font-weight:700;font-size:16px">{price}</div>
      <div style="color:#666;font-size:12px;margin-top:4px">{extra}</div>
    </div>"""


def product_grid(item_ids, info, extras=None, key_prefix="g"):
    extras = extras or {}
    rows = [item_ids[i:i + CARDS_PER_ROW] for i in range(0, len(item_ids), CARDS_PER_ROW)]
    for ri, row in enumerate(rows):
        cols = st.columns(CARDS_PER_ROW)
        for ci, iid in enumerate(row):
            with cols[ci]:
                st.markdown(card_html(iid, info[iid], extras.get(iid, "")),
                            unsafe_allow_html=True)
                if st.button("View", key=f"{key_prefix}-{ri}-{ci}-{iid}"):
                    st.session_state.page = ("product", iid)
                    st.rerun()


def recommend_for_user(model, u_idx, train_items, val_items, n, item_index):
    scores = model.score_all_items(u_idx).astype(float)
    excluded = train_items.get(u_idx, set()) | val_items.get(u_idx, set())
    if excluded:
        scores[list(excluded)] = -np.inf
    order = np.argsort(-scores, kind="stable")[:n]
    return [str(item_index[i]) for i in order if scores[i] > -np.inf]


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ElectroRec — thesis store demo", layout="wide")

ratings, train, val, test, info = load_data()
models = load_models(train)
hybrid = get_hybrid(models)

user_ids = np.sort(ratings["user_id"].unique())
item_ids = np.sort(ratings["item_id"].unique())
item_pos = {str(x): k for k, x in enumerate(item_ids)}
user_to_idx = {u: k for k, u in enumerate(user_ids)}
train_items = {int(u): set(g["i"].astype(int)) for u, g in train.groupby("u")}
val_items = {int(u): set(g["i"].astype(int)) for u, g in val.groupby("u")}
avg_rating = train.groupby("i")["rating"].mean()

if "page" not in st.session_state:
    st.session_state.page = ("home",)
if "uid" not in st.session_state:
    st.session_state.uid = str(user_ids[0])

# header
hc1, hc2, hc3 = st.columns([3, 4, 2])
with hc1:
    if st.button("🏠 ElectroRec", key="home-btn"):
        st.session_state.page = ("home",)
        st.rerun()
with hc2:
    q = st.text_input("Search products", placeholder="headphones, cable, charger…",
                      label_visibility="collapsed", key="search-q")
    if q and q != st.session_state.get("last_q"):
        st.session_state.last_q = q
        st.session_state.page = ("search", q)
        st.rerun()
with hc3:
    uid = st.text_input("Shopper (user id)", value=st.session_state.uid,
                        label_visibility="collapsed", key="uid-input")
    if uid in user_to_idx:
        st.session_state.uid = uid
u_idx = user_to_idx.get(st.session_state.uid, 0)

page = st.session_state.page

if page[0] == "home":
    st.header("Recommended for you")
    st.caption(f"Hybrid model (α={ALPHA}) — personalized for shopper "
               f"{st.session_state.uid}")
    recs = recommend_for_user(hybrid, u_idx, train_items, val_items, 8, item_ids)
    extras = {}
    for iid in recs:
        r = float(models["svd"].predict_rating(u_idx, item_pos[iid]))
        extras[iid] = f'<span style="color:#e67e22">{stars(r)}</span> predicted {r:.1f}'
    product_grid(recs, info, extras, "home-rec")

    st.header("Popular right now")
    st.caption("Most loved across all shoppers (popularity baseline)")
    pop_recs = recommend_for_user(models["popularity"], u_idx, train_items,
                                  val_items, 8, item_ids)
    extras = {}
    for iid in pop_recs:
        r = float(avg_rating.get(item_pos[iid], 0))
        extras[iid] = f'<span style="color:#e67e22">{stars(r)}</span> {r:.1f} avg'
    product_grid(pop_recs, info, extras, "home-pop")

    with st.expander("How the recommendations work (thesis results)"):
        import json
        sp = ROOT / "logs" / "eval_summary.json"
        if sp.is_file():
            payload = json.loads(sp.read_text())
            rows = []
            for name, metrics in payload["summary"].items():
                row = {"model": name}
                for m, agg in metrics.items():
                    row[m] = f"{agg['mean']:.4f} ± {agg['ci95']:.4f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Mean ± 95% CI over 5 seeds, 17,422 test users. "
                       "Full methodology: thesis/methodology.md")

elif page[0] == "search":
    query = page[1].lower()
    hits = [iid for iid, d in info.items() if query in d["title"].lower()][:24]
    st.header(f'Results for "{page[1]}" ({len(hits)} shown)')
    if hits:
        product_grid(hits, info, key_prefix="search")
    else:
        st.write("No products matched. Try 'cable', 'battery' or 'headphone'.")

elif page[0] == "product":
    iid = page[1]
    d = info[iid]
    pos = item_pos[iid]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""<img src="{d['image']}"
            onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
            style="width:100%;max-height:400px;object-fit:contain"/>""",
            unsafe_allow_html=True)
    with c2:
        st.header(d["title"])
        if d["price"]:
            st.subheader(f":red[{price_str(d['price'])}]")
        pred = float(models["svd"].predict_rating(u_idx, pos))
        avg = float(avg_rating.get(pos, 0))
        st.write(f"You'd probably rate it: **{pred:.1f}** "
                 f'<span style="color:#e67e22">{stars(pred)}</span>',
                 unsafe_allow_html=True)
        st.write(f"Average rating: **{avg:.1f}** "
                 f'<span style="color:#e67e22">{stars(avg)}</span>',
                 unsafe_allow_html=True)
        if d["desc"]:
            st.write(d["desc"][:500] + ("…" if len(d["desc"]) > 500 else ""))

    st.header("Similar items")
    sim = models["itemknn"].sim_topk.tocsr()
    s, e = sim.indptr[pos], sim.indptr[pos + 1]
    nbrs = sim.indices[s:e]
    vals = sim.data[s:e]
    if len(nbrs):
        st.caption("Items closest to this one (item-based collaborative filtering)")
        top = nbrs[np.argsort(-vals, kind="stable")[:8]]
        sim_ids = [str(item_ids[i]) for i in top if i != pos][:8]
    else:
        # cold item: no co-ratings -> fall back to content similarity
        st.caption("Items with the closest descriptions (content similarity)")
        emb = models["content"].item_embeddings
        sims = emb @ emb[pos]
        sims[pos] = -np.inf
        top = np.argsort(-sims, kind="stable")[:8]
        sim_ids = [str(item_ids[i]) for i in top]
    product_grid(sim_ids, info, key_prefix="sim")

    st.header("Shoppers also liked")
    st.caption("Hybrid picks for you in the same neighbourhood")
    more = recommend_for_user(hybrid, u_idx, train_items, val_items, 12, item_ids)
    more = [x for x in more if x != iid][:8]
    product_grid(more, info, key_prefix="also")
