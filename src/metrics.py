class RankingMetrics:
    @staticmethod
    def recall_at_k(model, val_targets, user_histories, user_seen_items, k=10):
        hits = 0
        total = 0

        for user_id, target_item in val_targets.items():
            history = user_histories.get(user_id, [])
            seen_items = user_seen_items.get(user_id, set())

            recommendations = model.recommend(
                user_id=user_id,
                user_history=history,
                seen_items=seen_items,
                k=k
            )

            if target_item in recommendations[:k]:
                hits += 1

            total += 1

        return hits / total if total > 0 else 0.0