"""Unit tests.

Everything here runs on tiny hand-made DataFrames. The tests that need the
real dataset (data/ratings.csv, data/metadata.csv) skip cleanly when the
files aren't there -- they never make up fake data instead. That silent
fake-data fallback was one of the Milestone 2 problems, so the skip is
deliberate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import encode_ids, load_metadata, load_ratings
from evaluator import make_splits, ranking_metrics_for_user
from hybrid_engine import (
    ContentEncoder,
    HybridModel,
    ItemKNNModel,
    PopularityModel,
    SVDModel,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "ratings.csv").is_file() and (DATA_DIR / "metadata.csv").is_file()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="real dataset not present")


def _tiny_ratings() -> pd.DataFrame:
    # 3 users x 4 items, strictly increasing timestamps per user.
    rows = []
    ts = 0
    ratings = {
        "u1": [("i1", 5.0), ("i2", 4.0), ("i3", 2.0), ("i4", 3.0)],
        "u2": [("i1", 4.0), ("i2", 5.0), ("i3", 1.0), ("i4", 2.0)],
        "u3": [("i1", 2.0), ("i2", 2.0), ("i3", 5.0), ("i4", 4.0)],
    }
    for u, pairs in ratings.items():
        for i, r in pairs:
            ts += 1
            rows.append((u, i, r, ts))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def _tiny_texts(n_items: int) -> list[str]:
    words = ["cable", "battery", "headphone", "charger", "speaker", "adapter"]
    return [" ".join(words[(k + j) % len(words)] for j in range(6)) for k in range(n_items)]


# --------------------------------------------------------------------------- #
# data_loader
# --------------------------------------------------------------------------- #
def test_missing_ratings_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ratings(tmp_path / "nope.csv")


def test_missing_metadata_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "nope.csv")


def test_encode_ids_is_deterministic_and_sorted():
    df = _tiny_ratings().sample(frac=1.0, random_state=0)  # shuffle rows
    enc1, m1, _ = encode_ids(df)
    enc2, m2, _ = encode_ids(df.sample(frac=1.0, random_state=1))
    assert m1 == m2  # mapping independent of row order
    assert list(m1) == sorted(m1)  # sorted order


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_split_rule_most_recent_is_test():
    df = _tiny_ratings()
    train, val, test, test_users = make_splits(df)
    # u1's most recent rating is (u1, i4) -> must be the test row.
    assert len(test) == 3 and len(val) == 3 and len(train) == 6
    t1 = test[test["user_id"] == "u1"].iloc[0]
    assert (t1["item_id"], t1["rating"]) == ("i4", 3.0)
    v1 = val[val["user_id"] == "u1"].iloc[0]
    assert (v1["item_id"], v1["rating"]) == ("i3", 2.0)
    assert set(test_users) == {0, 1, 2}
    # No overlap between partitions.
    triples = lambda d: set(zip(d["user_id"], d["item_id"]))
    assert triples(train).isdisjoint(triples(val))
    assert triples(train).isdisjoint(triples(test))
    assert triples(val).isdisjoint(triples(test))


def test_split_users_with_fewer_than_3_ratings_go_all_to_train():
    df = _tiny_ratings()
    extra = pd.DataFrame([("u9", "i1", 5.0, 999), ("u9", "i2", 4.0, 1000)],
                         columns=["user_id", "item_id", "rating", "timestamp"])
    df = pd.concat([df, extra], ignore_index=True)
    train, val, test, _ = make_splits(df)
    assert set(test[test["user_id"] == "u9"].index) == set()
    assert set(val[val["user_id"] == "u9"].index) == set()
    assert len(train[train["user_id"] == "u9"]) == 2


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _encoded_tiny():
    # make_splits returns frames that already carry deterministic integer
    # u/i codes (see data_loader.encode_ids).
    df = _tiny_ratings()
    train, _, _, _ = make_splits(df)
    return train, df["user_id"].nunique(), df["item_id"].nunique()


def test_popularity_predicts_item_mean():
    train, n_users, n_items = _encoded_tiny()
    m = PopularityModel().fit(train, n_users, n_items)
    # i1 train ratings: u1=5, u2=4 (u3's i1 rating went to... check manually via frame)
    sub = train[train["i"] == 0]
    assert m.predict_rating(0, 0) == pytest.approx(sub["rating"].mean())


def test_itemknn_predictions_in_range_and_deterministic():
    train, n_users, n_items = _encoded_tiny()
    m1 = ItemKNNModel(k=2, seed=7).fit(train, n_users, n_items)
    m2 = ItemKNNModel(k=2, seed=7).fit(train, n_users, n_items)
    p1 = m1.predict_ratings(np.array([[0, 0], [1, 2]]))
    p2 = m2.predict_ratings(np.array([[0, 0], [1, 2]]))
    assert np.all((p1 >= 1.0) & (p1 <= 5.0))
    assert np.allclose(p1, p2)


def test_svd_predictions_in_range_and_deterministic():
    train, n_users, n_items = _encoded_tiny()
    kw = dict(n_factors=4, n_epochs=3, seed=11)
    m1 = SVDModel(**kw).fit(train, n_users, n_items)
    m2 = SVDModel(**kw).fit(train, n_users, n_items)
    p1 = m1.predict_ratings(np.array([[0, 1], [2, 3]]))
    assert np.all((p1 >= 1.0) & (p1 <= 5.0))
    assert np.allclose(p1, m2.predict_ratings(np.array([[0, 1], [2, 3]])))


def test_svd_predict_ratings_matches_formula():
    # predict_ratings must equal mu + bu + bi + pu.qi per pair (this caught
    # a real bug where a wrong einsum subscript summed over the batch).
    train, n_users, n_items = _encoded_tiny()
    m = SVDModel(n_factors=4, n_epochs=3, seed=11).fit(train, n_users, n_items)
    pairs = np.array([[0, 1], [2, 3], [1, 0]])
    got = m.predict_ratings(pairs)
    for (u, i), g in zip(pairs, got):
        want = np.clip(m.mu + m.bu[u] + m.bi[i] + m.pu[u] @ m.qi[i], 1.0, 5.0)
        assert g == pytest.approx(want)
    # ... and agree with score_all_items for the same user.
    s = m.score_all_items(1)
    assert s[0] == pytest.approx(m.predict_rating(1, 0))
    assert s[3] == pytest.approx(m.predict_rating(1, 3))


def test_content_encoder_produces_unit_embeddings():
    train, n_users, n_items = _encoded_tiny()
    enc = ContentEncoder(n_epochs=2, batch_size=4, seed=5).fit(
        train, n_users, n_items, _tiny_texts(n_items))
    norms = np.linalg.norm(enc.item_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    s = enc.score_all_items(0)
    assert s.shape == (n_items,)
    assert np.all((s >= -1.0) & (s <= 1.0))


def test_hybrid_alpha_endpoints_match_components():
    train, n_users, n_items = _encoded_tiny()
    svd = SVDModel(n_factors=4, n_epochs=3, seed=3).fit(train, n_users, n_items)
    content = ContentEncoder(n_epochs=2, batch_size=4, seed=3).fit(
        train, n_users, n_items, _tiny_texts(n_items))
    h1 = HybridModel(svd, content, alpha=1.0)
    h0 = HybridModel(svd, content, alpha=0.0)
    cf = HybridModel._minmax(svd.score_all_items(0).astype(float))
    co = HybridModel._minmax(content.score_all_items(0).astype(float))
    assert np.allclose(h1.score_all_items(0), cf)
    assert np.allclose(h0.score_all_items(0), co)


def test_no_hardcoded_metrics_metrics_depend_on_data():
    """Regression test for the Milestone 2 problem: metrics must be computed,
    not typed in. If the validation ratings change, the RMSE has to change."""
    from evaluator import evaluate_models
    train, n_users, n_items = _encoded_tiny()
    _, val, _, _ = make_splits(_tiny_ratings())
    pop = PopularityModel().fit(train, n_users, n_items)
    res = evaluate_models({"popularity": pop}, train, val, n_items)
    rmse = res["popularity"]["rmse"]
    assert 0.0 < rmse < 4.0
    # Perturb one validation rating -> the metric must change.
    val2 = val.copy()
    val2.loc[val2.index[0], "rating"] = 1.0 if val2.loc[val2.index[0], "rating"] > 3 else 5.0
    res2 = evaluate_models({"popularity": pop}, train, val2, n_items)
    assert res2["popularity"]["rmse"] != pytest.approx(rmse)


def test_ranking_metrics_known_case():
    # Perfect ranking of a single relevant item at rank 1.
    ndcg, prec, rec = ranking_metrics_for_user(np.array([7, 3, 5]), {7})
    assert ndcg == pytest.approx(1.0)
    assert prec == pytest.approx(1 / 5)
    assert rec == pytest.approx(1.0)
    # No relevant items -> recall undefined (None), ndcg 0.
    ndcg, prec, rec = ranking_metrics_for_user(np.array([7, 3, 5]), set())
    assert rec is None and ndcg == 0.0


# --------------------------------------------------------------------------- #
# data-dependent (skipped cleanly when data is absent)
# --------------------------------------------------------------------------- #
@needs_data
def test_real_data_loads_and_splits():
    ratings = load_ratings(DATA_DIR / "ratings.csv")
    metadata = load_metadata(DATA_DIR / "metadata.csv")
    assert len(ratings) > 0 and len(metadata) > 0
    train, val, test, test_users = make_splits(ratings)
    assert len(test) == len(test_users) > 0
    assert len(train) > len(val)


@needs_data
def test_real_data_popularity_rmse_computed():
    from evaluator import evaluate_models
    ratings = load_ratings(DATA_DIR / "ratings.csv")
    train, val, _, _ = make_splits(ratings)
    enc, _, _ = encode_ids(ratings)
    pop = PopularityModel().fit(train, enc["u"].nunique(), enc["i"].nunique())
    res = evaluate_models({"popularity": pop}, train, val.head(200),
                          enc["i"].nunique())
    assert np.isfinite(res["popularity"]["rmse"])


# --------------------------------------------------------------------------- #
# persistence (save/load round-trips)
# --------------------------------------------------------------------------- #
def test_popularity_save_load_roundtrip(tmp_path):
    train, n_users, n_items = _encoded_tiny()
    m = PopularityModel().fit(train, n_users, n_items)
    p = tmp_path / "pop.npz"
    m.save(p)
    m2 = PopularityModel.load(p)
    assert np.allclose(m2.score_all_items(1), m.score_all_items(1))
    assert m2.predict_rating(0, 2) == m.predict_rating(0, 2)


def test_itemknn_save_load_roundtrip(tmp_path):
    train, n_users, n_items = _encoded_tiny()
    m = ItemKNNModel(k=2, seed=7).fit(train, n_users, n_items)
    p = tmp_path / "knn.npz"
    m.save(p)
    m2 = ItemKNNModel.load(p, train)
    assert np.allclose(m2.score_all_items(0), m.score_all_items(0))
    assert np.allclose(m2.predict_ratings(np.array([[0, 1], [2, 3]])),
                       m.predict_ratings(np.array([[0, 1], [2, 3]])))


def test_svd_save_load_roundtrip(tmp_path):
    train, n_users, n_items = _encoded_tiny()
    m = SVDModel(n_factors=4, n_epochs=3, seed=11).fit(train, n_users, n_items)
    p = tmp_path / "svd.npz"
    m.save(p)
    m2 = SVDModel.load(p)
    assert np.allclose(m2.score_all_items(2), m.score_all_items(2))
    assert m2.n_factors == 4


def test_content_save_load_roundtrip(tmp_path):
    train, n_users, n_items = _encoded_tiny()
    m = ContentEncoder(n_epochs=2, batch_size=4, seed=5).fit(
        train, n_users, n_items, _tiny_texts(n_items))
    p = tmp_path / "content.pt"
    m.save(p)
    m2 = ContentEncoder.load(p)
    assert np.allclose(m2.score_all_items(0), m.score_all_items(0))
    assert m2.item_embeddings.shape == m.item_embeddings.shape


# --------------------------------------------------------------------------- #
# hybrid + metric helpers
# --------------------------------------------------------------------------- #
def test_hybrid_alpha_out_of_range_raises():
    train, n_users, n_items = _encoded_tiny()
    svd = SVDModel(n_factors=4, n_epochs=2, seed=1).fit(train, n_users, n_items)
    content = ContentEncoder(n_epochs=1, batch_size=4, seed=1).fit(
        train, n_users, n_items, _tiny_texts(n_items))
    with pytest.raises(ValueError):
        HybridModel(svd, content, alpha=1.5)
    with pytest.raises(ValueError):
        HybridModel(svd, content, alpha=-0.1)


def test_hybrid_minmax_constant_vector_gives_zeros():
    z = HybridModel._minmax(np.array([3.0, 3.0, 3.0]))
    assert np.all(z == 0.0)
    mm = HybridModel._minmax(np.array([1.0, 2.0, 3.0]))
    assert mm[0] == pytest.approx(0.0) and mm[-1] == pytest.approx(1.0)


def test_topk_tie_break_goes_to_lower_index():
    from evaluator import _topk_ordered
    top = _topk_ordered(np.array([5.0, 5.0, 5.0, 1.0]), 2)
    assert list(top) == [0, 1]


def test_topk_ignores_masked_items():
    from evaluator import _topk_ordered
    scores = np.array([9.0, 5.0, 7.0])
    scores[0] = -np.inf  # how the evaluator marks excluded items
    top = _topk_ordered(scores, 2)
    assert 0 not in top
    assert list(top) == [2, 1]


def test_dcg_known_values():
    from evaluator import _dcg
    assert _dcg(np.array([])) == 0.0
    # single relevant item at rank 1: (2^1 - 1) / log2(2) = 1
    assert _dcg(np.array([1.0])) == pytest.approx(1.0)
    # relevant at rank 2 only: 1 / log2(3)
    assert _dcg(np.array([0.0, 1.0])) == pytest.approx(1.0 / np.log2(3))


def test_evaluate_models_returns_ranking_only_for_hybrid():
    from evaluator import evaluate_models
    train, n_users, n_items = _encoded_tiny()
    _, val, _, _ = make_splits(_tiny_ratings())
    svd = SVDModel(n_factors=4, n_epochs=2, seed=2).fit(train, n_users, n_items)
    content = ContentEncoder(n_epochs=1, batch_size=4, seed=2).fit(
        train, n_users, n_items, _tiny_texts(n_items))
    hybrid = HybridModel(svd, content, alpha=0.7)
    res = evaluate_models({"hybrid": hybrid, "svd": svd}, train, val, n_items)
    assert "rmse" not in res["hybrid"]  # hybrid outputs a score, not a rating
    assert "rmse" in res["svd"]
    for name in ("hybrid", "svd"):
        for metric in ("ndcg@5", "precision@5", "recall@5"):
            assert np.isfinite(res[name][metric])


# --------------------------------------------------------------------------- #
# data_loader validation
# --------------------------------------------------------------------------- #
def test_load_ratings_rejects_bad_columns(tmp_path):
    p = tmp_path / "ratings.csv"
    pd.DataFrame({"user_id": ["a"], "nope": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ratings(p)


def test_load_ratings_rejects_out_of_range(tmp_path):
    p = tmp_path / "ratings.csv"
    pd.DataFrame({"user_id": ["a"], "item_id": ["b"], "rating": [6.0],
                  "timestamp": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ratings(p)


def test_load_ratings_rejects_nulls(tmp_path):
    p = tmp_path / "ratings.csv"
    pd.DataFrame({"user_id": ["a"], "item_id": ["b"], "rating": [None],
                  "timestamp": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ratings(p)


def test_load_metadata_fills_missing_text(tmp_path):
    p = tmp_path / "meta.csv"
    pd.DataFrame({"item_id": ["b"], "title": [None],
                  "description": ["x"]}).to_csv(p, index=False)
    df = load_metadata(p)
    assert df.loc[0, "title"] == ""


def test_load_metadata_keeps_optional_storefront_columns(tmp_path):
    p = tmp_path / "meta.csv"
    pd.DataFrame({
        "item_id": ["b1", "b2"],
        "title": ["t1", "t2"],
        "description": ["d1", "d2"],
        "image_url": ["http://img/1.jpg", None],
        "price": ["9.99", None],
        "category": ["Cases", None],
    }).to_csv(p, index=False)
    df = load_metadata(p)
    assert df.loc[0, "image_url"] == "http://img/1.jpg"
    assert df.loc[1, "image_url"] == ""  # NaN -> "", never "nan"
    assert df.loc[1, "price"] == ""
    assert df.loc[0, "category"] == "Cases"


def test_load_metadata_works_without_optional_columns(tmp_path):
    # any catalog CSV works: only item_id + title required, optionals "" 
    p = tmp_path / "meta.csv"
    pd.DataFrame({"item_id": ["b"], "title": ["t"]}).to_csv(p, index=False)
    df = load_metadata(p)
    for c in ("description", "image_url", "price", "category"):
        assert c in df.columns and df.loc[0, c] == ""


# --------------------------------------------------------------------------- #
# more data_loader coverage
# --------------------------------------------------------------------------- #
def test_sha256_of_file(tmp_path):
    from data_loader import sha256_of_file
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    import hashlib
    assert sha256_of_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_load_all_loads_both(tmp_path):
    from data_loader import load_all
    rp = tmp_path / "ratings.csv"
    mp = tmp_path / "meta.csv"
    _tiny_ratings().to_csv(rp, index=False)
    pd.DataFrame({"item_id": ["i1"], "title": ["t"], "description": ["d"]}
                 ).to_csv(mp, index=False)
    ratings, meta = load_all(rp, mp)
    assert len(ratings) == 12 and len(meta) == 1


# --------------------------------------------------------------------------- #
# metric helper edge cases
# --------------------------------------------------------------------------- #
def test_topk_edge_cases():
    from evaluator import _topk_ordered, _dcg
    assert len(_topk_ordered(np.array([1.0, 2.0]), 0)) == 0
    # k bigger than n -> all items, ordered
    assert list(_topk_ordered(np.array([1.0, 3.0, 2.0]), 10)) == [1, 2, 0]
    assert _dcg(np.array([1.0, 0.0, 1.0])) > _dcg(np.array([0.0, 1.0, 1.0]))


def test_itemknn_unknown_user_falls_back_to_item_mean():
    train, n_users, n_items = _encoded_tiny()
    m = ItemKNNModel(k=2, seed=7).fit(train, n_users, n_items)
    # user 999 was never seen: prediction = item mean, clipped
    assert m.predict_rating(999, 0) == pytest.approx(
        np.clip(m.item_mean[0], 1.0, 5.0))
    s = m.score_all_items(999)
    assert np.allclose(s, np.clip(m.item_mean, 1.0, 5.0))


# --------------------------------------------------------------------------- #
# tuning + full experiment driver on tiny data
# --------------------------------------------------------------------------- #
def _write_tiny_csvs(tmp_path):
    rp = tmp_path / "ratings.csv"
    mp = tmp_path / "metadata.csv"
    _tiny_ratings().to_csv(rp, index=False)
    pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i4"],
        "title": ["cable fast", "battery pack", "headphone loud", "charger mini"],
        "description": ["a", "b", "c", "d"],
    }).to_csv(mp, index=False)
    return tmp_path


def test_tuning_grid_search_picks_best_cell(tmp_path):
    from tuning import run_grid_search
    import json
    data_dir = _write_tiny_csvs(tmp_path)
    payload = run_grid_search(svd_factors_grid=[2], alpha_grid=[0.0, 0.5, 1.0],
                              seed=42, log_dir=tmp_path / "logs", data_dir=data_dir)
    assert len(payload["grid_results"]) == 3
    best = payload["best"]
    assert best["svd_factors"] == 2
    # best cell really is the max ndcg@5 in the grid
    top = max(r["ndcg@5"] for r in payload["grid_results"])
    assert best["ndcg@5"] == pytest.approx(top)
    assert (tmp_path / "logs" / "grid_search.json").is_file()


def test_run_experiment_writes_logs(tmp_path):
    from evaluator import run_experiment
    import json
    data_dir = _write_tiny_csvs(tmp_path)
    payload = run_experiment(seeds=[7], svd_factors=2, alpha=0.5,
                             log_dir=tmp_path / "logs", data_dir=data_dir)
    for f in ("eval_seeds.json", "eval_summary.json", "summary.md"):
        assert (tmp_path / "logs" / f).is_file()
    assert payload["summary"]["hybrid"]["ndcg@5"]["n_seeds"] == 1
    assert payload["config"]["svd_factors"] == 2
    # summary.md is human-readable and mentions the models
    md = (tmp_path / "logs" / "summary.md").read_text()
    assert "hybrid" in md and "svd" in md


def test_run_experiment_resumes_from_partial_checkpoint(tmp_path, capsys):
    from evaluator import run_experiment
    import json, shutil
    data_dir = _write_tiny_csvs(tmp_path)
    log_dir = tmp_path / "logs"
    run_experiment(seeds=[7], svd_factors=2, alpha=0.5,
                   log_dir=log_dir, data_dir=data_dir)
    # Fake a partial checkpoint from the finished run, then re-run: the seed
    # must be picked up, not recomputed.
    finished = json.loads((log_dir / "eval_seeds.json").read_text())
    with open(log_dir / "eval_seeds_partial.json", "w") as f:
        json.dump({"config": finished["config"],
                   "per_seed": finished["per_seed"]}, f)
    payload = run_experiment(seeds=[7], svd_factors=2, alpha=0.5,
                             log_dir=log_dir, data_dir=data_dir)
    assert "Resuming with 1 finished seed(s)" in capsys.readouterr().out
    assert payload["summary"]["hybrid"]["ndcg@5"]["n_seeds"] == 1


def test_run_experiment_ignores_checkpoint_for_other_config(tmp_path, capsys):
    from evaluator import run_experiment
    import json
    data_dir = _write_tiny_csvs(tmp_path)
    log_dir = tmp_path / "logs"
    (log_dir).mkdir(parents=True, exist_ok=True)
    with open(log_dir / "eval_seeds_partial.json", "w") as f:
        json.dump({"config": {"svd_factors": 999, "hybrid_alpha": 0.5,
                              "data": {"ratings_sha256": "nope"}},
                   "per_seed": {}}, f)
    run_experiment(seeds=[7], svd_factors=2, alpha=0.5,
                   log_dir=log_dir, data_dir=data_dir)
    assert "different config; ignoring it" in capsys.readouterr().out


def test_itemknn_topk_truncation_keeps_best(tmp_path):
    train, n_users, n_items = _encoded_tiny()
    m = ItemKNNModel(k=1, seed=7).fit(train, n_users, n_items)
    sim = m.sim_topk.tocsr()
    for i in range(n_items):
        assert sim.indptr[i + 1] - sim.indptr[i] <= 1
