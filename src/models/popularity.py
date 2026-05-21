class PopularityRecommender:
    def __init__(self):
        self.popular_items = []

    def fit(self, train_df):
        item_counts = train_df["item_id"].value_counts()
        self.popular_items = item_counts.index.tolist()
        return self

    def recommend(self, user_id, user_history=None, seen_items=None, k=10):
        if seen_items is None:
            seen_items = set()

        recommendations = []

        for item_id in self.popular_items:
            if item_id not in seen_items:
                recommendations.append(item_id)

            if len(recommendations) == k:
                break

        return recommendations