"""The recommendation models.

Every model is trained only on the training split it gets passed in fit().
None of them ever sees validation or test data -- that's enforced by the
evaluator, which only hands them the train partition.

The models:
* Popularity -- predicts an item's mean training rating (falls back to the
  global mean for unseen items); ranks by mean rating, ties broken by how
  many ratings the item has.
* ItemKNN -- item-based collaborative filtering I wrote from scratch: cosine
  similarity on mean-centred training ratings, with shrinkage and only the
  top-50 neighbours kept per item. I went custom instead of Surprise's
  KNNBasic so the whole thing is deterministic and there's no compiled
  dependency to build.
* SVD -- biased matrix factorisation (global mean + user/item biases + latent
  factors) trained with SGD, also from scratch for the same reason.
* ContentEncoder -- the "dense neural embeddings" part: a small PyTorch MLP
  turns TF-IDF(title + description) into a 32-dim item embedding, a learned
  user embedding is dotted with it, and the net is trained with MSE on the
  training ratings (fixed seed, CPU, deterministic settings). For ranking,
  the content score is the cosine between the item embedding and the user's
  profile, which is just the mean embedding of the items they rated >= 4.0
  in training.
* HybridModel -- score(u,i) = alpha * norm(CF) + (1-alpha) * norm(content).
  Both parts are min-max normalised to [0,1] per user before blending, so
  alpha actually means something as a blend weight. Note the hybrid outputs
  a blended similarity score, not a rating, so RMSE/MAE are meaningless for
  it -- the evaluator only reports ranking metrics for the hybrid (this was
  one of the Milestone 2 mistakes: RMSE on a cosine score).

Nothing in this file contains metric values -- every number in the report is
computed by src/evaluator.py from what these models output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RATING_MIN, RATING_MAX = 1.0, 5.0
RELEVANCE_THRESHOLD = 4.0  # fixed relevance threshold for ranking metrics


# --------------------------------------------------------------------------- #
# Popularity baseline
# --------------------------------------------------------------------------- #
class PopularityModel:
    """Predict item mean rating; rank by mean rating (count tie-break)."""

    name = "popularity"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "PopularityModel":
        self.n_users, self.n_items = n_users, n_items
        grp = train.groupby("i")["rating"]
        self.item_mean = np.full(n_items, np.nan)
        self.item_count = np.zeros(n_items)
        means = grp.mean()
        counts = grp.size()
        self.item_mean[means.index.to_numpy()] = means.to_numpy()
        self.item_count[counts.index.to_numpy()] = counts.to_numpy()
        self.global_mean = float(train["rating"].mean())
        self.item_mean = np.where(np.isnan(self.item_mean), self.global_mean, self.item_mean)
        return self

    def predict_rating(self, u: int, i: int) -> float:
        return float(np.clip(self.item_mean[i], RATING_MIN, RATING_MAX))

    def predict_ratings(self, pairs: np.ndarray) -> np.ndarray:
        return np.clip(self.item_mean[pairs[:, 1].astype(int)], RATING_MIN, RATING_MAX)

    def score_all_items(self, u: int) -> np.ndarray:
        # For ranking: higher mean first; the tiny count/index terms just make
        # ties deterministic instead of depending on sort internals.
        return self.item_mean + 1e-9 * self.item_count - 1e-12 * np.arange(self.n_items)

    def save(self, path) -> None:
        np.savez(path, item_mean=self.item_mean, item_count=self.item_count,
                 global_mean=self.global_mean, n_users=self.n_users, n_items=self.n_items)

    @classmethod
    def load(cls, path) -> "PopularityModel":
        z = np.load(path)
        m = cls()
        m.item_mean = z["item_mean"]
        m.item_count = z["item_count"]
        m.global_mean = float(z["global_mean"])
        m.n_users, m.n_items = int(z["n_users"]), int(z["n_items"])
        return m


# --------------------------------------------------------------------------- #
# Item-based KNN (custom cosine item-item CF)
# --------------------------------------------------------------------------- #
class ItemKNNModel:
    """Item-item CF with cosine similarity, shrinkage and top-k truncation."""

    name = "itemknn"

    def __init__(self, k: int = 50, shrinkage: float = 100.0, seed: int = 42):
        self.k = k
        self.shrinkage = shrinkage
        self.seed = seed

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "ItemKNNModel":
        self.n_users, self.n_items = n_users, n_items
        ratings = train["rating"].to_numpy(dtype=np.float64)
        rows = train["i"].to_numpy()
        cols = train["u"].to_numpy()

        item_sum = np.bincount(rows, weights=ratings, minlength=n_items)
        item_cnt = np.bincount(rows, minlength=n_items).astype(np.float64)
        self.item_mean = np.divide(item_sum, item_cnt, out=np.full(n_items, ratings.mean()), where=item_cnt > 0)
        self.global_mean = float(ratings.mean())

        centered = ratings - self.item_mean[rows]
        X = sp.csr_matrix((centered, (rows, cols)), shape=(n_items, n_users))
        Xn = normalize(X, norm="l2", axis=1)  # now dot products = cosines

        # How many users rated each item pair (needed for shrinkage below).
        B = (X != 0).astype(np.float64)
        common = (B @ B.T).tocsr()

        sim = (Xn @ Xn.T).tocsr()
        # Shrink similarities computed from few co-ratings toward zero:
        # sim * common / (common + shrinkage). Sparse element-wise multiply
        # because sim and common can differ in sparsity pattern (pairs whose
        # cosine is exactly 0 get dropped from sim).
        common.data = common.data / (common.data + self.shrinkage)
        sim = sim.multiply(common).tocsr()

        # Keep only the top-k neighbours per item (never the item itself).
        # Sorted by similarity desc, item index asc so it's reproducible.
        sim.setdiag(0.0)
        sim.eliminate_zeros()
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        sim_csr = sim.tocsr()
        for i in range(n_items):
            s, e = sim_csr.indptr[i], sim_csr.indptr[i + 1]
            nbrs = sim_csr.indices[s:e]
            vals = sim_csr.data[s:e]
            if len(nbrs) > self.k:
                order = np.lexsort((nbrs, -vals))[: self.k]
                nbrs, vals = nbrs[order], vals[order]
            else:
                order = np.lexsort((nbrs, -vals))
                nbrs, vals = nbrs[order], vals[order]
            indices.extend(nbrs.tolist())
            data.extend(vals.tolist())
            indptr.append(len(indices))
        self.sim_topk = sp.csr_matrix(
            (np.array(data), np.array(indices, dtype=np.int64), np.array(indptr)),
            shape=(n_items, n_items),
        )
        # Per-user centred rating vectors for fast scoring.
        self._user_centered = {
            int(u): (g["i"].to_numpy(), (g["rating"].to_numpy() - self.item_mean[g["i"].to_numpy()]))
            for u, g in train.groupby("u")
        }
        return self

    def _predict_one(self, u: int, i: int) -> float:
        base = self.item_mean[i]
        if u not in self._user_centered:
            return float(np.clip(base, RATING_MIN, RATING_MAX))
        js, centred = self._user_centered[u]
        s, e = self.sim_topk.indptr[i], self.sim_topk.indptr[i + 1]
        nbrs = self.sim_topk.indices[s:e]
        sims = self.sim_topk.data[s:e]
        mask = np.isin(nbrs, js)
        if not np.any(mask):
            return float(np.clip(base, RATING_MIN, RATING_MAX))
        nbrs, sims = nbrs[mask], sims[mask]
        lut = dict(zip(js.tolist(), centred.tolist()))
        num = sum(sims[t] * lut[int(nbrs[t])] for t in range(len(nbrs)))
        den = float(np.abs(sims).sum())
        if den == 0.0:
            return float(np.clip(base, RATING_MIN, RATING_MAX))
        return float(np.clip(base + num / den, RATING_MIN, RATING_MAX))

    def predict_rating(self, u: int, i: int) -> float:
        return self._predict_one(u, i)

    def predict_ratings(self, pairs: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(int(u), int(i)) for u, i in pairs])

    def score_all_items(self, u: int) -> np.ndarray:
        scores = self.item_mean.copy()
        if u in self._user_centered:
            js, centred = self._user_centered[u]
            vec = np.zeros(self.n_items)
            vec[js] = centred
            num = self.sim_topk @ vec
            den = np.abs(self.sim_topk) @ (vec != 0).astype(float)
            upd = den > 0
            scores[upd] += num[upd] / den[upd]
        return np.clip(scores, RATING_MIN, RATING_MAX)

    def save(self, path) -> None:
        s = self.sim_topk.tocsr()
        np.savez(path, item_mean=self.item_mean, global_mean=self.global_mean,
                 sim_data=s.data, sim_indices=s.indices, sim_indptr=s.indptr,
                 shape=np.array(s.shape), k=self.k, shrinkage=self.shrinkage,
                 seed=self.seed, n_users=self.n_users, n_items=self.n_items)

    @classmethod
    def load(cls, path, train: pd.DataFrame) -> "ItemKNNModel":
        # Needs the train frame to rebuild the per-user centred rating vectors.
        z = np.load(path)
        m = cls(k=int(z["k"]), shrinkage=float(z["shrinkage"]), seed=int(z["seed"]))
        m.n_users, m.n_items = int(z["n_users"]), int(z["n_items"])
        m.item_mean = z["item_mean"]
        m.global_mean = float(z["global_mean"])
        m.sim_topk = sp.csr_matrix(
            (z["sim_data"], z["sim_indices"], z["sim_indptr"]),
            shape=tuple(int(x) for x in z["shape"]))
        m._user_centered = {
            int(u): (g["i"].to_numpy(), g["rating"].to_numpy() - m.item_mean[g["i"].to_numpy()])
            for u, g in train.groupby("u")
        }
        return m


# --------------------------------------------------------------------------- #
# Biased matrix factorisation (FunkSVD) trained with SGD
# --------------------------------------------------------------------------- #
class SVDModel:
    """Biased matrix factorisation trained with SGD.

    Predicts rating = global mean + user bias + item bias + p_u . q_i.
    Plain SGD, one pass over a seeded shuffle per epoch, so a fixed seed
    gives bit-identical results every run.
    """

    name = "svd"

    def __init__(self, n_factors: int = 50, lr: float = 0.005, reg: float = 0.02,
                 n_epochs: int = 20, seed: int = 42, verbose: bool = False):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.verbose = verbose

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "SVDModel":
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = n_users, n_items
        u = train["u"].to_numpy()
        i = train["i"].to_numpy()
        r = train["rating"].to_numpy(dtype=np.float64)
        n = len(train)

        self.mu = float(r.mean())
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.pu = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        self.qi = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))

        lr, reg = self.lr, self.reg
        for epoch in range(self.n_epochs):
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  svd epoch {epoch + 1}/{self.n_epochs}", flush=True)
            order = rng.permutation(n)
            for idx in order:
                uu, ii, rr = u[idx], i[idx], r[idx]
                err = rr - (self.mu + self.bu[uu] + self.bi[ii] + self.pu[uu] @ self.qi[ii])
                self.bu[uu] += lr * (err - reg * self.bu[uu])
                self.bi[ii] += lr * (err - reg * self.bi[ii])
                pu_old = self.pu[uu].copy()
                self.pu[uu] += lr * (err * self.qi[ii] - reg * self.pu[uu])
                self.qi[ii] += lr * (err * pu_old - reg * self.qi[ii])
        return self

    def predict_rating(self, u: int, i: int) -> float:
        return float(np.clip(self.mu + self.bu[u] + self.bi[i] + self.pu[u] @ self.qi[i],
                             RATING_MIN, RATING_MAX))

    def predict_ratings(self, pairs: np.ndarray) -> np.ndarray:
        uu = pairs[:, 0].astype(int)
        ii = pairs[:, 1].astype(int)
        # NOTE: both batch dims must share the same label ("uf,uf->u").
        # "uf,if->u" would sum over the whole batch -- a subtle einsum trap.
        preds = self.mu + self.bu[uu] + self.bi[ii] + np.einsum("uf,uf->u", self.pu[uu], self.qi[ii])
        return np.clip(preds, RATING_MIN, RATING_MAX)

    def score_all_items(self, u: int) -> np.ndarray:
        return np.clip(self.mu + self.bu[u] + self.bi + self.pu[u] @ self.qi.T,
                       RATING_MIN, RATING_MAX)

    def save(self, path) -> None:
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi, pu=self.pu, qi=self.qi,
                 n_factors=self.n_factors, lr=self.lr, reg=self.reg,
                 n_epochs=self.n_epochs, seed=self.seed,
                 n_users=self.n_users, n_items=self.n_items)

    @classmethod
    def load(cls, path) -> "SVDModel":
        z = np.load(path)
        m = cls(n_factors=int(z["n_factors"]), lr=float(z["lr"]),
                reg=float(z["reg"]), n_epochs=int(z["n_epochs"]), seed=int(z["seed"]))
        m.mu = float(z["mu"])
        m.bu, m.bi, m.pu, m.qi = z["bu"], z["bi"], z["pu"], z["qi"]
        m.n_users, m.n_items = int(z["n_users"]), int(z["n_items"])
        return m


# --------------------------------------------------------------------------- #
# Dense neural content embeddings (PyTorch)
# --------------------------------------------------------------------------- #
class _ItemTower(nn.Module):
    """MLP: TF-IDF(title + description) -> 32-d dense item embedding."""

    def __init__(self, in_dim: int, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContentEncoder:
    """Dense item embeddings from a small neural net.

    The item tower is an MLP: TF-IDF(title + description) -> 128 -> ReLU ->
    32-dim embedding. There's also a learned 32-dim embedding per user, and
    the model predicts rating = global mean + dot(user emb, item emb),
    trained with MSE on the training ratings only. Fixed seed, CPU,
    deterministic PyTorch settings, so it's reproducible.

    After training, the item embeddings are what the hybrid uses: the content
    score for a user is the cosine between each item's embedding and the
    user's profile (mean embedding of the items they rated >= 4.0 in
    training; if they never rated anything that high, mean over all their
    training items).
    """

    name = "content"

    def __init__(self, emb_dim: int = 32, tfidf_features: int = 512,
                 lr: float = 0.01, n_epochs: int = 10, batch_size: int = 4096,
                 seed: int = 42):
        self.emb_dim = emb_dim
        self.tfidf_features = tfidf_features
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int,
            item_texts: list[str]) -> "ContentEncoder":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        self.n_users, self.n_items = n_users, n_items
        # float32 and a single dense copy shared with torch (no duplicate
        # array): 42384 items x 512 features x 4 bytes is about 87 MB, which
        # matters on a small machine.
        vectorizer = TfidfVectorizer(max_features=self.tfidf_features,
                                     stop_words="english", dtype=np.float32)
        X_sparse = vectorizer.fit_transform(item_texts)
        Xt = torch.from_numpy(X_sparse.toarray())
        del X_sparse
        self._vectorizer = vectorizer

        item_tower = _ItemTower(Xt.shape[1], self.emb_dim)
        self._tower = item_tower
        self._in_dim = Xt.shape[1]
        user_emb = nn.Embedding(n_users, self.emb_dim)
        nn.init.normal_(user_emb.weight, std=0.1)
        mu = float(train["rating"].mean())

        opt = torch.optim.Adam(list(item_tower.parameters()) + list(user_emb.parameters()),
                               lr=self.lr)
        loss_fn = nn.MSELoss()
        uu = torch.from_numpy(train["u"].to_numpy()).long()
        ii = torch.from_numpy(train["i"].to_numpy()).long()
        rr = torch.from_numpy(train["rating"].to_numpy(dtype=np.float32))
        n = len(train)
        gen = torch.Generator().manual_seed(self.seed)

        item_tower.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, self.batch_size):
                b = perm[s: s + self.batch_size]
                e = item_tower(Xt[ii[b]])
                pred = mu + (user_emb(uu[b]) * e).sum(dim=1)
                loss = loss_fn(pred, rr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()

        item_tower.eval()
        with torch.no_grad():
            E = item_tower(Xt).numpy()
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.item_embeddings = E / norms  # L2-normalised dense embeddings

        # User profile = mean embedding of items rated >= 4.0 in TRAIN.
        liked = train[train["rating"] >= RELEVANCE_THRESHOLD]
        profiles = np.zeros((n_users, self.emb_dim))
        counts = np.zeros(n_users)
        for u, g in liked.groupby("u"):
            profiles[int(u)] = self.item_embeddings[g["i"].to_numpy()].mean(axis=0)
            counts[int(u)] = len(g)
        # Fallback for users with no liked train items: mean over all train items.
        missing = counts == 0
        if missing.any():
            for u, g in train.groupby("u"):
                if missing[int(u)]:
                    profiles[int(u)] = self.item_embeddings[g["i"].to_numpy()].mean(axis=0)
        pn = np.linalg.norm(profiles, axis=1, keepdims=True)
        pn[pn == 0.0] = 1.0
        self.user_profiles = profiles / pn
        return self

    def content_scores(self, u: int) -> np.ndarray:
        """Cosine similarity between user profile and every item embedding."""
        return self.item_embeddings @ self.user_profiles[u]

    def score_all_items(self, u: int) -> np.ndarray:
        return self.content_scores(u)

    def save(self, path) -> None:
        torch.save({
            "tower": self._tower.state_dict(),
            "in_dim": self._in_dim,
            "emb_dim": self.emb_dim,
            "vectorizer": self._vectorizer,
            "item_embeddings": self.item_embeddings,
            "user_profiles": self.user_profiles,
            "n_users": self.n_users,
            "n_items": self.n_items,
        }, path)

    @classmethod
    def load(cls, path) -> "ContentEncoder":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(emb_dim=d["emb_dim"])
        m._tower = _ItemTower(d["in_dim"], d["emb_dim"])
        m._tower.load_state_dict(d["tower"])
        m._tower.eval()
        m._in_dim = d["in_dim"]
        m._vectorizer = d["vectorizer"]
        m.item_embeddings = d["item_embeddings"]
        m.user_profiles = d["user_profiles"]
        m.n_users, m.n_items = d["n_users"], d["n_items"]
        return m


# --------------------------------------------------------------------------- #
# Weighted hybrid
# --------------------------------------------------------------------------- #
class HybridModel:
    """Weighted blend of the CF and content scores.

    score = alpha * norm(cf) + (1 - alpha) * norm(content), where each side is
    min-max normalised to [0,1] per user first. Without that normalisation
    alpha wouldn't be comparable across the two score scales, so the per-user
    normalisation is what makes the weight meaningful.
    """

    name = "hybrid"

    def __init__(self, cf_model, content_encoder: ContentEncoder, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.cf_model = cf_model
        self.content_encoder = content_encoder
        self.alpha = alpha

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def score_all_items(self, u: int) -> np.ndarray:
        cf = self._minmax(self.cf_model.score_all_items(u).astype(float))
        co = self._minmax(self.content_encoder.score_all_items(u).astype(float))
        return self.alpha * cf + (1.0 - self.alpha) * co
