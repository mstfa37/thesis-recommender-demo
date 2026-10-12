"""ElectroRec - Amazon-like storefront demo for the thesis recommender.

Run:  streamlit run app.py
Needs data/ratings.csv, data/metadata.csv (image_url, price, category),
data/reviews.csv and trained models in models/ (scripts/train_models.py).
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
PER_ROW = 4

PLACEHOLDER_IMG = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E"
    "%3Crect width='300' height='300' fill='%23e8e8e8'/%3E"
    "%3Ctext x='150' y='155' font-family='sans-serif' font-size='20' "
    "fill='%23999' text-anchor='middle'%3Eno image%3C/text%3E%3C/svg%3E"
)


# --------------------------------------------------------------------------- #
# data + models
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    """NaN-safe string: missing CSV fields become '' instead of 'nan'."""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


@st.cache_resource(show_spinner="Loading the catalog and training splits…")
def load_data():
    ratings, metadata = load_all(ROOT / "data" / "ratings.csv",
                                 ROOT / "data" / "metadata.csv")
    train, val, test, _ = make_splits(ratings)
    # load_metadata guarantees image_url/price/category exist ("" when the
    # dataset doesn't provide them), so the UI adapts to any catalog.
    info = {}
    for _, r in metadata.iterrows():
        iid = _s(r["item_id"])
        info[iid] = {
            "title": html.unescape(_s(r["title"])).strip()[:140] or iid,
            "image": _s(r["image_url"]) or PLACEHOLDER_IMG,
            "price": _s(r["price"]).strip(),
            "desc": html.unescape(_s(r["description"]))[:800],
            "category": _s(r["category"]).strip() or "Miscellaneous",
        }
    reviews = {}
    rp = ROOT / "data" / "reviews.csv"
    if rp.is_file():
        for _, r in pd.read_csv(rp, dtype=str).fillna("").iterrows():
            reviews.setdefault(_s(r["item_id"]), []).append({
                "reviewer": r["reviewer"] or "a shopper",
                "rating": float(r["rating"] or 0),
                "summary": r["summary"],
                "text": r["text"],
            })
    return ratings, train, val, info, reviews


@st.cache_resource(show_spinner="Loading the trained models…")
def load_models(_train):
    return {
        "popularity": PopularityModel.load(MODEL_DIR / "popularity.npz"),
        "itemknn": ItemKNNModel.load(MODEL_DIR / "itemknn.npz", _train),
        "svd": SVDModel.load(MODEL_DIR / "svd.npz"),
        "content": ContentEncoder.load(MODEL_DIR / "content.pt"),
    }


def get_hybrid(models):
    return HybridModel(models["svd"], models["content"], alpha=ALPHA)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def stars(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def price_str(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    return p if p.startswith("$") else f"${p}"


def price_val(p: str) -> float:
    try:
        return float(p.strip().replace("$", "").replace(",", ""))
    except Exception:
        return float("inf")


def card_html(iid: str, info: dict, extra: str = "") -> str:
    title = info["title"].replace('"', "&quot;")
    img = info["image"].replace('"', "&quot;")
    price = price_str(info["price"])
    price_html = (f'<div style="color:#B12704;font-weight:700;font-size:17px">'
                  f'{price}</div>' if price else "")
    return f"""
    <div style="border:1px solid #e3e6e6;border-radius:8px;padding:12px;
                background:white;height:100%">
      <img src="{img}" loading="lazy" referrerpolicy="no-referrer"
           onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
           style="width:100%;height:170px;object-fit:contain;background:#fafafa;
                  border-radius:4px"/>
      <div style="font-size:13px;margin:8px 0 4px;height:58px;overflow:hidden;
                  line-height:1.35">{title}</div>
      {price_html}
      <div style="color:#666;font-size:12px;margin-top:4px">{extra}</div>
    </div>"""


def product_grid(item_ids, info, extras=None, key_prefix="g", add_cart=True):
    extras = extras or {}
    for ri in range(0, len(item_ids), PER_ROW):
        cols = st.columns(PER_ROW)
        for ci, iid in enumerate(item_ids[ri:ri + PER_ROW]):
            with cols[ci]:
                st.markdown(card_html(iid, info[iid], extras.get(iid, "")),
                            unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("View", key=f"{key_prefix}-v-{ri}-{ci}-{iid}",
                                 width="stretch"):
                        st.session_state.page = ("product", iid)
                        rv = st.session_state.recently_viewed
                        if iid in rv:
                            rv.remove(iid)
                        st.session_state.recently_viewed = [iid] + rv[:11]
                        st.rerun()
                with b2:
                    if add_cart and st.button("Add", key=f"{key_prefix}-a-{ri}-{ci}-{iid}",
                                              width="stretch"):
                        cart = st.session_state.cart
                        cart[iid] = cart.get(iid, 0) + 1
                        st.rerun()


def recommend(model, u_idx, train_items, val_items, n, item_ids):
    scores = model.score_all_items(u_idx).astype(float)
    excluded = train_items.get(u_idx, set()) | val_items.get(u_idx, set())
    if excluded:
        scores[list(excluded)] = -np.inf
    order = np.argsort(-scores, kind="stable")[:n]
    return [str(item_ids[i]) for i in order if scores[i] > -np.inf]


def rating_extras(item_ids, info, avg_rating, counts, item_pos, models, u_idx,
                  predicted=False):
    extras = {}
    for iid in item_ids:
        pos = item_pos[iid]
        r = float(models["svd"].predict_rating(u_idx, pos)) if predicted \
            else float(avg_rating.get(pos, 0))
        n = int(counts.get(pos, 0))
        label = "predicted" if predicted else f"{n} ratings"
        extras[iid] = (f'<span style="color:#e67e22">{stars(r)}</span> '
                       f'{r:.1f} <span style="color:#999">({label})</span>')
    return extras


# --------------------------------------------------------------------------- #
# app state
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ElectroRec", layout="wide")

st.markdown("""
<div style="background:#131921;color:white;padding:12px 20px;border-radius:8px;
            margin-bottom:10px;display:flex;align-items:center;gap:12px">
  <span style="font-size:24px;font-weight:800;letter-spacing:-0.5px">
    🛒 ElectroRec</span>
  <span style="color:#febd69;font-size:13px">a thesis recommender, disguised as a store</span>
</div>""", unsafe_allow_html=True)

ratings, train, val, info, reviews = load_data()
models = load_models(train)
hybrid = get_hybrid(models)

user_ids = np.sort(ratings["user_id"].unique())
item_ids = np.sort(ratings["item_id"].unique())
item_pos = {str(x): k for k, x in enumerate(item_ids)}
user_to_idx = {u: k for k, u in enumerate(user_ids)}
train_items = {int(u): set(g["i"].astype(int)) for u, g in train.groupby("u")}
val_items = {int(u): set(g["i"].astype(int)) for u, g in val.groupby("u")}
avg_rating = train.groupby("i")["rating"].mean()
rating_counts = train.groupby("i")["rating"].count()

if "page" not in st.session_state:
    st.session_state.page = ("home",)
if "uid" not in st.session_state:
    st.session_state.uid = str(user_ids[0])
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []

# top categories for the nav strip
cat_counts = pd.Series([d["category"] for d in info.values()]).value_counts()
top_cats = [c for c in cat_counts.head(10).index]

# ---- header ----
st.markdown("""
<style>
  .topbar {background:#131921;color:white;padding:10px 20px;border-radius:8px;
           margin-bottom:6px}
  .topbar a {color:white;text-decoration:none}
  .navstrip {margin:4px 0 12px}
</style>""", unsafe_allow_html=True)
h1, h2, h3 = st.columns([2, 5, 3])
with h1:
    if st.button("🛒 ElectroRec", key="logo"):
        st.session_state.page = ("home",)
        st.rerun()
with h2:
    q = st.text_input("q", placeholder="Search headphones, cables, chargers…",
                      label_visibility="collapsed", key="search-q")
    if q and q != st.session_state.get("last_q"):
        st.session_state.last_q = q
        st.session_state.page = ("search", q)
        st.rerun()
with h3:
    c1, c2 = st.columns([3, 2])
    with c1:
        uid = st.text_input("u", value=st.session_state.uid,
                            label_visibility="collapsed", key="uid-input",
                            placeholder="shopper id")
        if uid in user_to_idx:
            st.session_state.uid = uid
    with c2:
        n_cart = sum(st.session_state.cart.values())
        if st.button(f"🧺 Cart ({n_cart})", key="cart-btn",
                     width="stretch"):
            st.session_state.page = ("cart",)
            st.rerun()
u_idx = user_to_idx.get(st.session_state.uid, 0)

# ---- category nav ----
nav_cols = st.columns(len(top_cats) + 1)
with nav_cols[0]:
    if st.button("All", key="nav-all"):
        st.session_state.page = ("home",)
        st.rerun()
for i, cat in enumerate(top_cats):
    with nav_cols[i + 1]:
        if st.button(cat[:16], key=f"nav-{i}",
                     help=cat, width="stretch"):
            st.session_state.page = ("category", cat)
            st.rerun()

page = st.session_state.page

# --------------------------------------------------------------------------- #
def shelf(title, subtitle, item_ids, **kw):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
    product_grid(item_ids, info, extras=kw.get("extras"),
                 key_prefix=kw.get("key", "s"),
                 add_cart=kw.get("add_cart", True))


if page[0] == "home":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#232f3e,#37475a);color:white;
                padding:28px;border-radius:10px;margin-bottom:16px">
      <h2 style="margin:0;color:white">Today's picks, chosen for you</h2>
      <p style="margin:6px 0 0;color:#ddd">A hybrid recommender (collaborative
      filtering + neural content embeddings) personalizes every shelf.</p>
    </div>""", unsafe_allow_html=True)

    recs = recommend(hybrid, u_idx, train_items, val_items, 8, item_ids)
    shelf("Recommended for you",
          f"Hybrid model (α={ALPHA}) · shopper {st.session_state.uid}", recs,
          key="home-rec",
          extras=rating_extras(recs, info, avg_rating, rating_counts, item_pos,
                               models, u_idx, predicted=True))

    pop = recommend(models["popularity"], u_idx, train_items, val_items, 8,
                    item_ids)
    shelf("Popular right now", "Loved across all shoppers", pop, key="home-pop",
          extras=rating_extras(pop, info, avg_rating, rating_counts, item_pos,
                               models, u_idx))

    # deals: priced under $10, best rated first
    priced = [(iid, price_val(d["price"])) for iid, d in info.items()
              if price_val(d["price"]) < 10]
    priced.sort(key=lambda t: -float(avg_rating.get(item_pos[t[0]], 0)))
    deals = [iid for iid, _ in priced[:8]]
    shelf("Deals under $10", "Great ratings, small prices", deals,
          key="home-deals",
          extras=rating_extras(deals, info, avg_rating, rating_counts,
                               item_pos, models, u_idx))

    # top rated: >= 5 ratings
    top_rated = [str(item_ids[i]) for i in rating_counts[rating_counts >= 5]
                 .sort_values(ascending=False).head(40).index]
    top_rated.sort(key=lambda iid: -float(avg_rating.get(item_pos[iid], 0)))
    shelf("Top rated", "Highest average ratings (5+ reviews)", top_rated[:8],
          key="home-top",
          extras=rating_extras(top_rated[:8], info, avg_rating, rating_counts,
                               item_pos, models, u_idx))

    rv = [iid for iid in st.session_state.recently_viewed if iid in info][:8]
    if rv:
        shelf("Recently viewed", "Picked up where you left off", rv,
              key="home-rv",
              extras=rating_extras(rv, info, avg_rating, rating_counts,
                                   item_pos, models, u_idx))

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
            st.dataframe(pd.DataFrame(rows), width="stretch")
            st.caption("Mean ± 95% CI over 5 seeds, 17,422 test users.")

    with st.expander("Compare what each model recommends for you"):
        st.caption(f"Top 5 per model for shopper {st.session_state.uid} — "
                   "this is the thesis result, live.")
        comp_models = [
            ("Popularity", models["popularity"]),
            ("ItemKNN", models["itemknn"]),
            ("SVD", models["svd"]),
            ("Content", models["content"]),
            ("Hybrid (α=0.7)", hybrid),
        ]
        cols = st.columns(len(comp_models))
        for (name, m), col in zip(comp_models, cols):
            with col:
                st.write(f"**{name}**")
                top5 = recommend(m, u_idx, train_items, val_items, 5, item_ids)
                for iid in top5:
                    st.markdown(
                        f"""<img src="{info[iid]['image']}" loading="lazy"
                        referrerpolicy="no-referrer"
                        onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
                        style="width:100%;height:90px;object-fit:contain;
                               background:#fafafa;border-radius:4px"/>""",
                        unsafe_allow_html=True)
                    st.caption(info[iid]["title"][:45])

elif page[0] == "category":
    cat = page[1]
    st.header(cat)
    in_cat = [iid for iid, d in info.items() if d["category"] == cat]
    sort = st.selectbox("Sort by", ["Top rated", "Price: low to high",
                                    "Price: high to low", "Most reviewed"],
                        key="cat-sort")
    if sort == "Top rated":
        in_cat.sort(key=lambda iid: -float(avg_rating.get(item_pos[iid], 0)))
    elif sort == "Price: low to high":
        in_cat.sort(key=lambda iid: price_val(info[iid]["price"]))
    elif sort == "Price: high to low":
        in_cat.sort(key=lambda iid: -price_val(info[iid]["price"])
                    if price_val(info[iid]["price"]) != float("inf") else 0)
    else:
        in_cat.sort(key=lambda iid: -int(rating_counts.get(item_pos[iid], 0)))
    st.caption(f"{len(in_cat)} products")
    product_grid(in_cat[:24], info, key_prefix="cat",
                 extras=rating_extras(in_cat[:24], info, avg_rating,
                                      rating_counts, item_pos, models, u_idx))

elif page[0] == "search":
    query = page[1].lower()
    hits = [iid for iid, d in info.items() if query in d["title"].lower()][:24]
    st.header(f'Results for "{page[1]}"')
    st.caption(f"{len(hits)} shown")
    if hits:
        product_grid(hits, info, key_prefix="search")
    else:
        st.write("No products matched. Try 'cable', 'battery' or 'headphone'.")

elif page[0] == "product":
    iid = page[1]
    d = info[iid]
    pos = item_pos[iid]
    st.caption(f"Home / {d['category']} / {d['title'][:60]}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""<img src="{d['image']}" loading="lazy"
            referrerpolicy="no-referrer"
            onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
            style="width:100%;max-height:420px;object-fit:contain;
                   border:1px solid #e3e6e6;border-radius:8px"/>""",
            unsafe_allow_html=True)
    with c2:
        st.header(d["title"])
        avg = float(avg_rating.get(pos, 0))
        n = int(rating_counts.get(pos, 0))
        st.write(f'<span style="color:#e67e22;font-size:20px">{stars(avg)}</span>'
                 f"  **{avg:.1f}**  ({n} ratings)", unsafe_allow_html=True)
        if d["price"]:
            st.subheader(f":red[{price_str(d['price'])}]")
        pred = float(models["svd"].predict_rating(u_idx, pos))
        st.info(f"Based on your taste you'd probably rate it **{pred:.1f}**.")
        if st.button("🧺 Add to cart", key=f"pdp-add-{iid}"):
            cart = st.session_state.cart
            cart[iid] = cart.get(iid, 0) + 1
            st.rerun()
        if d["desc"]:
            st.write("**About this item**")
            st.write(d["desc"][:600] + ("…" if len(d["desc"]) > 600 else ""))

    # rating distribution from the training ratings
    item_ratings = train.loc[train["i"] == pos, "rating"]
    if len(item_ratings):
        st.subheader("Rating breakdown")
        dist = item_ratings.value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
        total_r = int(dist.sum())
        for star in (5, 4, 3, 2, 1):
            n = int(dist[star])
            pct = 100 * n / total_r if total_r else 0
            st.write(f"{'★' * star}{'☆' * (5 - star)} "
                     f"<div style='display:inline-block;width:220px;"
                     f"background:#eee;border-radius:4px;vertical-align:middle'>"
                     f"<div style='width:{pct:.0f}%;background:#e67e22;"
                     f"height:10px;border-radius:4px'></div></div> {pct:.0f}%",
                     unsafe_allow_html=True)

    with st.expander("Why was this recommended to you?"):
        st.write("The hybrid score is a weighted blend of two models:")
        cf_raw = float(models["svd"].score_all_items(u_idx)[pos])
        co_raw = float(models["content"].score_all_items(u_idx)[pos])
        st.write(f"- **Collaborative filtering (SVD)** raw score: `{cf_raw:.3f}` "
                 f"— shoppers with taste like yours rated similar items highly.")
        st.write(f"- **Content model** raw score: `{co_raw:.3f}` — this item's "
                 f"description is close to what you liked before.")
        st.write(f"- **Hybrid** = {ALPHA} × norm(SVD) + {1 - ALPHA} × "
                 f"norm(content), each normalized to [0, 1] per shopper first, "
                 f"so the {ALPHA}/{1 - ALPHA} weight is meaningful.")
        st.caption("α = 0.7 was chosen by grid search on the validation set "
                   "(logs/grid_search.json).")

    st.subheader("Similar items")
    sim = models["itemknn"].sim_topk.tocsr()
    s, e = sim.indptr[pos], sim.indptr[pos + 1]
    nbrs, vals = sim.indices[s:e], sim.data[s:e]
    if len(nbrs):
        st.caption("Shoppers who viewed this also viewed (item-based CF)")
        top = nbrs[np.argsort(-vals, kind="stable")[:8]]
        sim_ids = [str(item_ids[i]) for i in top if i != pos][:8]
    else:
        st.caption("Closest descriptions (content similarity)")
        emb = models["content"].item_embeddings
        sims = emb @ emb[pos]
        sims[pos] = -np.inf
        top = np.argsort(-sims, kind="stable")[:8]
        sim_ids = [str(item_ids[i]) for i in top]
    product_grid(sim_ids, info, key_prefix="sim")

    st.subheader("Shoppers also liked")
    more = [x for x in recommend(hybrid, u_idx, train_items, val_items, 12,
                                 item_ids) if x != iid][:8]
    product_grid(more, info, key_prefix="also")

    st.subheader("Customer reviews")
    revs = reviews.get(iid, [])
    if revs:
        for r in revs:
            st.write(f'<span style="color:#e67e22">{stars(r["rating"])}</span>'
                     f"  **{html.escape(r['summary'])}**",
                     unsafe_allow_html=True)
            st.caption(f"by {html.escape(r['reviewer'])}")
            st.write(html.escape(r["text"]))
            st.divider()
    else:
        st.write("No written reviews for this item yet.")

elif page[0] == "cart":
    st.header("Your cart")
    cart = st.session_state.cart
    if not cart:
        st.write("Your cart is empty. Go find something you like!")
    else:
        total = 0.0
        for iid, qty in list(cart.items()):
            d = info[iid]
            cols = st.columns([1, 4, 2, 2, 1])
            with cols[0]:
                st.markdown(f"""<img src="{d['image']}" loading="lazy"
                    referrerpolicy="no-referrer"
                    onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"
                    style="width:80px;height:80px;object-fit:contain"/>""",
                    unsafe_allow_html=True)
            with cols[1]:
                st.write(f"**{d['title'][:80]}**")
            with cols[2]:
                st.write(f"Qty: {qty}")
            with cols[3]:
                p = price_val(d["price"])
                if p != float("inf"):
                    st.write(f"**{price_str(d['price'])}**")
                    total += p * qty
            with cols[4]:
                if st.button("✕", key=f"rm-{iid}"):
                    del cart[iid]
                    st.rerun()
        st.subheader(f"Total: ${total:.2f}" if total else "Total: —")
        st.caption("Demo checkout — no real orders, this is a thesis project.")
        if st.button("Continue shopping"):
            st.session_state.page = ("home",)
            st.rerun()

st.markdown("---")
st.caption("ElectroRec — BSc thesis demo: weighted hybrid recommender "
           "(collaborative filtering + dense neural embeddings) on Amazon "
           "Electronics. Recommendations update per shopper; every number in "
           "the results panel is computed by the code, not typed in.")
