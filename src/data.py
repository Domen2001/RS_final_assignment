import pandas as pd
from collections import defaultdict


class ColumnDetector:
    @staticmethod
    def detect_column(df, candidates, column_type):
        lower_to_original = {col.lower(): col for col in df.columns}

        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                return lower_to_original[candidate.lower()]

        raise ValueError(
            f"Could not detect {column_type} column. "
            f"Available columns: {list(df.columns)}"
        )

    @staticmethod
    def detect_interaction_columns(df):
        user_col = ColumnDetector.detect_column(
            df,
            candidates=[
                "user_id", "userid", "user", "uid", "customer_id", "reviewerid"
            ],
            column_type="user"
        )

        item_col = ColumnDetector.detect_column(
            df,
            candidates=[
                "item_id", "itemid", "item", "iid", "product_id", "asin"
            ],
            column_type="item"
        )

        timestamp_col = ColumnDetector.detect_column(
            df,
            candidates=[
                "timestamp", "time", "unixReviewTime", "date", "datetime"
            ],
            column_type="timestamp"
        )

        return user_col, item_col, timestamp_col


class InteractionDataLoader:
    def __init__(self, train_path):
        self.train_path = train_path

    def load_train(self):
        df = pd.read_csv(self.train_path)

        user_col, item_col, timestamp_col = ColumnDetector.detect_interaction_columns(df)

        print("Detected columns:")
        print(f"User column:      {user_col}")
        print(f"Item column:      {item_col}")
        print(f"Timestamp column: {timestamp_col}")

        df = df[[user_col, item_col, timestamp_col]].copy()
        df.columns = ["user_id", "item_id", "timestamp"]

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["user_id", "item_id", "timestamp"])

        # Keep latest duplicate user-item interaction
        df = (
            df.sort_values(["user_id", "item_id", "timestamp"])
              .drop_duplicates(["user_id", "item_id"], keep="last")
        )

        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        return df


class TemporalLeaveOneOutSplitter:
    def split(self, df):
        train_parts = []
        val_targets = {}

        for user_id, user_df in df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp")

            if len(user_df) < 2:
                continue

            history = user_df.iloc[:-1]
            target = user_df.iloc[-1]

            train_parts.append(history)
            val_targets[user_id] = target["item_id"]

        if not train_parts:
            raise ValueError("No users with at least 2 interactions were found.")

        train_split = pd.concat(train_parts, ignore_index=True)

        return train_split, val_targets


class UserHistoryBuilder:
    @staticmethod
    def build_user_histories(df):
        user_histories = {}

        for user_id, user_df in df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp")
            user_histories[user_id] = user_df["item_id"].tolist()

        return user_histories

    @staticmethod
    def build_user_seen_items(df):
        user_seen_items = defaultdict(set)

        for row in df.itertuples(index=False):
            user_seen_items[row.user_id].add(row.item_id)

        return user_seen_items