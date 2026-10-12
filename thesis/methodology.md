# Methodology

## 1. Dataset

The experiments use the Amazon Electronics 5-core snapshot from McAuley's
lab (UCSD), 1,689,188 reviews from 1996–2014, obtained via the official
Stanford SNAP mirror. Each review carries a reviewer ID, a product ASIN, a
star rating (1–5) and a Unix timestamp; product titles and descriptions come
from the matching metadata file. To keep runtimes reasonable the pipeline
keeps a deterministic ~9% subsample of users: a user is included iff
`int(md5(reviewerID).hexdigest(), 16) % 100 < 9`, with all of their reviews.
The hash-based rule needs no stored state and keeps whole user histories,
which the per-user split requires. The final working set has 154,413
ratings from 17,422 users over 42,384 items, each item with a title and a
description.

## 2. Train / validation / test split

For each user the ratings are sorted newest-first (ties broken by item id,
so the ordering is fully deterministic) and assigned as follows: the newest
rating goes to the **test** set, the second newest to **validation**, and
the rest to **train**. Users with fewer than 3 ratings would go entirely to
train; this case does not occur in the 5-core data, but the rule is
implemented and documented. Every model trains **only** on the train
partition (119,569 ratings). The test partition (17,422 ratings, one per
user) and the validation partition (17,422) are never seen during training.
At test time each user's candidate items exclude their train *and*
validation items; during tuning on validation, candidates exclude the train
items. The test item is never used as recommendation context.

## 3. Models

**Popularity (baseline).** Predicts an item's mean training rating (global
mean for unseen items) and ranks by mean rating, ties broken by rating
count.

**ItemKNN (baseline).** Item-based collaborative filtering implemented from
scratch: cosine similarity on mean-centred training ratings, shrinkage
(100.0) of similarities computed from few co-ratings toward zero, and only
the top-50 neighbours kept per item. A custom implementation was chosen
over library code so that training is fully deterministic with no compiled
dependencies.

**SVD.** Biased matrix factorisation (global mean + user/item biases +
latent factors) trained with SGD (learning rate 0.005, regularisation
0.02, 20 epochs), also implemented from scratch for the same reason.

**Content model (dense neural embeddings).** A small PyTorch MLP maps
TF-IDF features of the product title + description (512 features) to a
32-dimensional item embedding. A learned 32-dimensional user embedding is
dotted with the item embedding and the network is trained with MSE on the
training ratings (fixed seed, CPU, deterministic settings). For ranking, the
content score of an item for a user is the cosine similarity between the
item embedding and the user's profile, defined as the mean embedding of the
items the user rated ≥ 4.0 in training (falling back to the mean over all
their training items).

**Hybrid.** `score(u,i) = α · norm(CF) + (1−α) · norm(content)`, where each
side is min-max normalised to [0,1] per user before blending; without this
normalisation the weight α would not be comparable across the two score
scales. The CF side is the SVD prediction. Since the hybrid outputs a
similarity score rather than a rating, RMSE/MAE are not reported for it.

## 4. Hyperparameter tuning

The hybrid weight α and the SVD factorisation size were tuned by grid
search on the validation split: SVD factors {20, 50, 100} × α
{0.0, 0.1, …, 1.0}, selecting by NDCG@5. The content encoder trains only
once since it does not depend on the SVD factors. The selected
configuration is **SVD factors = 20, α = 0.7**; larger factorisations did
not improve validation NDCG@5.

## 5. Evaluation protocol

Rating-prediction models (Popularity, ItemKNN, SVD) are scored with
**RMSE** and **MAE** over the test triples. All models are scored with the
ranking metrics **NDCG@5**, **Precision@5** and **Recall@5**, using a fixed
relevance threshold of rating ≥ 4.0. Precision@5 and NDCG@5 are averaged
over all evaluated users; Recall@5 is averaged over users with at least one
relevant item, since recall is undefined otherwise.

The full pipeline — split, train on the train partition, evaluate on the
test partition — is run once per seed in [42, 123, 2026, 7, 99], and results
are reported as mean ± 95% confidence interval (1.96·std/√n) across seeds.
Every model follows the same code path on every seed. Each run writes raw
JSON logs (`logs/eval_seeds.json`, `logs/eval_summary.json`,
`logs/grid_search.json`) recording the git commit, dataset row counts and
SHA-256 hashes, the split rule, seeds and hyperparameters, so every
reported number is traceable to a committed log.

## 6. Demo user interface

A Streamlit storefront (`app.py`, "ElectroRec") demonstrates the system end
to end as an Amazon-like shop: a home page with a personalized
"Recommended for you" shelf (hybrid model) and a "Popular right now" shelf,
full-text product search, and product pages showing the item's image, price,
predicted rating (SVD), average rating, a "Similar items" shelf (item-based
neighbours, content-similarity fallback for cold items) and a "Shoppers also
liked" shelf (hybrid). Product images and prices come from the raw Amazon
metadata (`scripts/add_product_media.py`). Trained models are saved to
`models/` after the first training run (`scripts/train_models.py`) so the UI
loads instantly on later starts.
