# Weighted Hybrid Recommendation Engine (Amazon Electronics)

My BSc Computer Science thesis project at ELTE. Supervisor: Mr. Md. Easin Arafat.

This is the corrected Milestone 3 version. Milestone 2 got flagged by my
supervisor because its results couldn't be reproduced from the code
(hardcoded metrics, train/test leakage, fake fallback data) — all of that is
fixed here, and the full story is in [REPORT.md](REPORT.md). Every number in
the report comes from committed code and traces back to a log file in `logs/`.

## What's in here

| model | what it does |
|---|---|
| `popularity` | baseline: predicts an item's mean training rating |
| `itemknn` | baseline: item-based CF, cosine similarity + shrinkage, top-50 neighbours (my own implementation) |
| `svd` | biased matrix factorisation trained with SGD (my own implementation) |
| `content` | dense neural item embeddings: small PyTorch MLP over TF-IDF(title+description), trained with MSE on train ratings |
| `hybrid` | `α · norm(CF) + (1−α) · norm(content)`, normalised per user so α means something; `hybrid_alpha0` / `hybrid_alpha1` are the pure-content / pure-CF ablations |

RMSE/MAE are only reported for the models that predict ratings (popularity,
itemknn, svd). Ranking metrics (NDCG@5, Precision@5, Recall@5) use a fixed
relevance cutoff of rating ≥ 4.0 for every model.

## Reproducing it

```bash
# 1. environment (Python 3.12, CPU-only torch)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

# 2. data: downloads the real Amazon Electronics 5-core snapshot and builds
#    data/ratings.csv + data/metadata.csv (deterministic, ~9% of users)
python scripts/download_data.py

# 3. tests (the ones needing the dataset skip cleanly if it's missing --
#    nothing fake is ever generated)
PYTHONPATH=src pytest -q tests/

# 4. grid search on the validation split
#    (SVD factors x alpha, picked by NDCG@5 -> logs/grid_search.json)
PYTHONPATH=src python src/tuning.py

# 5. multi-seed evaluation on the test split
#    (means + 95% CIs -> logs/eval_seeds.json, logs/eval_summary.json, logs/summary.md)
PYTHONPATH=src python src/evaluator.py --svd-factors 20 --alpha 0.7
```

## Demo UI

```bash
# train once and save the models (a few minutes), then:
streamlit run app.py
```

Pick a user id and a model and it shows the items the user liked in
training plus the model's top-N recommendations with product titles.
The first start trains everything and saves it to `models/` (gitignored);
later starts load the saved files instantly. The Results tab shows the
test numbers from `logs/eval_summary.json`.

## Thesis text

`thesis/methodology.md` and `thesis/results.md` are the methodology and
results sections; `REPORT.md` (not committed) is the full corrected
evaluation report.

## Layout

```
app.py              Streamlit demo UI (recommendations + results tabs)
src/
  data_loader.py    loads data/*.csv; raises FileNotFoundError if missing (never fakes data)
  hybrid_engine.py  Popularity / ItemKNN / SVD / ContentEncoder / HybridModel (+ save/load)
  evaluator.py      per-user train/val/test split, multi-seed eval, JSON logs
  tuning.py         grid search (SVD factors x alpha) on the validation split
scripts/
  download_data.py  builds data/ratings.csv + data/metadata.csv from the real snapshot
  train_models.py   trains everything once, saves to models/ for the UI
tests/
  test_engine.py    unit tests (skip cleanly without data; no mocks, no fake data)
thesis/             methodology.md + results.md (thesis sections)
logs/               committed raw logs: download_log.json, grid_search.json,
                    eval_seeds.json, eval_summary.json, summary.md
data/               ratings.csv, metadata.csv (built by scripts/download_data.py)
models/             saved trained models for the UI (gitignored, built by train_models.py)
```

## The evaluation setup, briefly

Per user, newest rating → test, second newest → validation, rest → train
(users with < 3 ratings go fully to train). Models train only on train. Test
covers all test users; candidates exclude the user's train+validation items.
REPORT.md has the full methodology.
