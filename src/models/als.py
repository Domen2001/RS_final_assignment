import os

# Helps avoid OpenBLAS slowdown warning on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import csr_matrix


class ALSRecommender:
    """
    Matrix factorization recommender using implicit ALS.

    This model learns:
    - a vector representation for each user
    - a vector representation for each item

    Then it recommends items whose vectors match the user's vector.
    """

    def __init__(
        self,
        factors=128,
        regularization=0.05,
        iterations=50,
        alpha=20.0,
        fallback_model=None,
        random_state=42
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.fallback_model = fallback_model
        self.random_state = random_state

        self.model = None

        self.user_to_idx = {}
        self.idx_to_user = {}

        self.item_to_idx = {}
        self.idx_to_item = {}

        self.user_item_matrix = None

    def fit(self, train_df):
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError(
                "The 'implicit' package is required for ALSRecommender. "
                "Install it with: pip install implicit"
            ) from exc

        print("Creating user and item ID mappings...")

        unique_users = train_df["user_id"].unique()
        unique_items = train_df["item_id"].unique()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}

        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}

        print(f"ALS users: {len(self.user_to_idx):,}")
        print(f"ALS items: {len(self.item_to_idx):,}")

        print("Building sparse user-item matrix...")

        user_indices = train_df["user_id"].map(self.user_to_idx).to_numpy()
        item_indices = train_df["item_id"].map(self.item_to_idx).to_numpy()

        values = np.ones(len(train_df), dtype=np.float32)

        self.user_item_matrix = csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(len(self.user_to_idx), len(self.item_to_idx))
        )

        confidence_matrix = (self.user_item_matrix * self.alpha).tocsr()

        print("Training ALS model...")

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state
        )

        # Important:
        # Modern implicit expects user-item matrix, shape = users x items.
        # Do NOT transpose here.
        self.model.fit(confidence_matrix)

        return self

    def recommend(self, user_id, user_history=None, seen_items=None, k=10):
        if seen_items is None:
            seen_items = set()

        if user_history is None:
            user_history = []

        if user_id not in self.user_to_idx:
            return self._fallback_recommend(user_id, user_history, seen_items, k)

        user_idx = self.user_to_idx[user_id]

        user_items = self.user_item_matrix[user_idx]

        item_indices, scores = self.model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=k * 10,
            filter_already_liked_items=True
        )

        recommendations = []

        for item_idx in item_indices:
            item_idx = int(item_idx)

            if item_idx not in self.idx_to_item:
                continue

            item_id = self.idx_to_item[item_idx]

            if item_id in seen_items:
                continue

            recommendations.append(item_id)

            if len(recommendations) == k:
                break

        if len(recommendations) < k:
            fallback_recs = self._fallback_recommend(
                user_id=user_id,
                user_history=user_history,
                seen_items=seen_items.union(set(recommendations)),
                k=k - len(recommendations)
            )

            recommendations.extend(fallback_recs)

        return recommendations[:k]

    def _fallback_recommend(self, user_id, user_history, seen_items, k):
        if self.fallback_model is None:
            return []

        return self.fallback_model.recommend(
            user_id=user_id,
            user_history=user_history,
            seen_items=seen_items,
            k=k
        )