import math
from collections import defaultdict, Counter


class ItemKNNRecommender:
    """
    Item-item co-occurrence recommender.

    Idea:
    - Items that occur in the same user histories are considered similar.
    - At prediction time, look at the user's recent items.
    - Recommend items that are similar to those recent items.
    """

    def __init__(
        self,
        max_history_items=20,
        max_neighbors_per_item=200,
        min_cooc_count=1,
        fallback_model=None
    ):
        self.max_history_items = max_history_items
        self.max_neighbors_per_item = max_neighbors_per_item
        self.min_cooc_count = min_cooc_count
        self.fallback_model = fallback_model

        self.item_popularity = Counter()
        self.item_neighbors = {}

    def fit(self, train_df):
        print("Building item popularity...")
        self.item_popularity = Counter(train_df["item_id"].tolist())

        print("Building user histories...")
        user_histories = (
            train_df.sort_values(["user_id", "timestamp"])
                    .groupby("user_id")["item_id"]
                    .apply(list)
                    .to_dict()
        )

        print("Building item-item co-occurrence matrix...")
        cooc = defaultdict(Counter)

        for _, history in user_histories.items():
            # Remove repeated items while preserving chronological order
            history = list(dict.fromkeys(history))

            if len(history) < 2:
                continue

            # Use only most recent items to reduce noise and computation
            recent_history = history[-self.max_history_items:]

            for i in range(len(recent_history)):
                item_i = recent_history[i]

                for j in range(i + 1, len(recent_history)):
                    item_j = recent_history[j]

                    distance = j - i

                    # Nearby items in the sequence receive stronger weight
                    weight = 1.0 / math.log2(2.0 + distance)

                    cooc[item_i][item_j] += weight
                    cooc[item_j][item_i] += weight

        print("Normalizing similarities and keeping top neighbors...")
        self.item_neighbors = {}

        for item_i, neighbors in cooc.items():
            scored_neighbors = []

            pop_i = self.item_popularity[item_i]

            for item_j, cooc_score in neighbors.items():
                if cooc_score < self.min_cooc_count:
                    continue

                pop_j = self.item_popularity[item_j]

                # Cosine-style normalization
                sim = cooc_score / math.sqrt(pop_i * pop_j)

                scored_neighbors.append((item_j, sim))

            scored_neighbors.sort(key=lambda x: x[1], reverse=True)

            self.item_neighbors[item_i] = scored_neighbors[:self.max_neighbors_per_item]

        print(f"Built neighbors for {len(self.item_neighbors):,} items.")

        return self

    def recommend(self, user_id, user_history=None, seen_items=None, k=10):
        if user_history is None:
            user_history = []

        if seen_items is None:
            seen_items = set()

        scores = defaultdict(float)

        recent_history = user_history[-self.max_history_items:]

        # Most recent items should matter more
        reversed_history = list(reversed(recent_history))

        for position, history_item in enumerate(reversed_history):
            recency_weight = 1.0 / math.log2(2.0 + position)

            neighbors = self.item_neighbors.get(history_item, [])

            for candidate_item, similarity in neighbors:
                if candidate_item in seen_items:
                    continue

                scores[candidate_item] += similarity * recency_weight

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = [item_id for item_id, _ in ranked_items[:k]]

        # Fill missing slots with popularity fallback
        if len(recommendations) < k and self.fallback_model is not None:
            fallback_recs = self.fallback_model.recommend(
                user_id=user_id,
                user_history=user_history,
                seen_items=seen_items.union(set(recommendations)),
                k=k - len(recommendations)
            )

            recommendations.extend(fallback_recs)

        return recommendations[:k]