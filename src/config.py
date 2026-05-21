from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"
    ITEM_META_PATH = DATA_DIR / "item_meta.csv"
    SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

    K = 10

    # ItemKNN settings
    MAX_HISTORY_ITEMS = 20
    MAX_NEIGHBORS_PER_ITEM = 200
    MIN_COOC_COUNT = 1

    # ALS settings
    ALS_FACTORS = 128
    ALS_REGULARIZATION = 0.05
    ALS_ITERATIONS = 50
    ALS_ALPHA = 20.0