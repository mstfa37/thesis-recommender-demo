# Results

All numbers are mean ± 95% CI over 5 seeds (42, 123, 2026, 7, 99),
17,422 test users. Full per-seed logs: `logs/eval_seeds.json`,
`logs/eval_summary.json`.

## 1. Tuning (validation split)

Grid: SVD factors {20, 50, 100} × α {0.0 … 1.0}, selected by NDCG@5.
Full grid: `logs/grid_search.json`.

| factors | α=0.0 (pure content) | α=0.7 | α=1.0 (pure CF) |
|---|---|---|---|
| 20 | 0.00090 | **0.00160** | 0.00120 |
| 50 | 0.00090 | 0.00133 | 0.00111 |
| 100 | 0.00090 | 0.00137 | 0.00112 |

The hybrid beats both of its components on their own (at 20 factors:
0.00160 vs 0.00090 for pure content and 0.00120 for pure SVD), and larger
factorisations do not help (20 ≥ 50 ≈ 100). Selected: **factors = 20,
α = 0.7**.

## 2. Test results

| model | RMSE | MAE | NDCG@5 | P@5 | R@5 |
|---|---|---|---|---|---|
| popularity | 1.3051 ± 0.0000 | 0.9283 ± 0.0000 | 0.0005 ± 0.0000 | 0.0001 ± 0.0000 | 0.0009 ± 0.0000 |
| itemknn | 1.3112 ± 0.0000 | 0.9296 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0001 ± 0.0000 |
| svd | **1.1727 ± 0.0003** | **0.8781 ± 0.0005** | 0.0010 ± 0.0002 | 0.0004 ± 0.0001 | 0.0022 ± 0.0004 |
| content | — | — | 0.0006 ± 0.0001 | 0.0002 ± 0.0000 | 0.0013 ± 0.0002 |
| **hybrid (α=0.7)** | — | — | **0.0014 ± 0.0001** | **0.0005 ± 0.0000** | **0.0030 ± 0.0002** |
| hybrid α=0.0 (pure content) | — | — | 0.0006 ± 0.0001 | 0.0002 ± 0.0000 | 0.0013 ± 0.0002 |
| hybrid α=1.0 (pure CF) | — | — | 0.0010 ± 0.0002 | 0.0004 ± 0.0001 | 0.0022 ± 0.0004 |

RMSE/MAE are defined only for the rating-prediction models (popularity,
itemknn, svd); the hybrid and pure-content models output similarity scores,
not ratings.

## 3. Interpretation

**SVD predicts ratings best.** Its RMSE of 1.1727 clearly beats the
baselines (1.3051 popularity, 1.3112 itemKNN): its star-rating predictions
are closer to the real ones.

**The hybrid ranks best.** On all three ranking metrics it outperforms both
of its parts alone, and the ablation rows make this direct: α=0.7
(NDCG@5 0.0014) vs pure content α=0.0 (0.0006) vs pure SVD α=1.0 (0.0010).
Blending the collaborative and content signals genuinely helps; neither
part alone reaches the hybrid's ranking quality.

**ItemKNN ranks poorly here** — worse than plain popularity. Its rating
predictions are reasonable (RMSE 1.3112), but with few ratings per item the
item neighbourhoods are noisy, so the personalised re-ranking adds more
noise than signal.

**On the scale of the numbers.** The absolute ranking values are small
because the protocol is demanding: one held-out item per user out of a
42,384-item catalog. Random ranking would give Precision@5 ≈ 0.00012; the
hybrid reaches 0.0005, roughly 4× better. The contribution of this
milestone is a sound, reproducible methodology rather than large absolute
scores.

## 4. Limitations

- The 9% user subsample keeps runtimes manageable but discards most of the
  snapshot; results describe this subsample, not the full 1.7M-review
  dataset.
- The content model uses TF-IDF + a small MLP instead of a pre-trained
  sentence encoder, trading semantic richness for full offline
  reproducibility.
- Ranking is evaluated against a single held-out item per user, which makes
  absolute metric values small and noisy; the relative ordering of models
  is the meaningful signal.
