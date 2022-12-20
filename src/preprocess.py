from pathlib import Path

import pandas as pd

from src.config import log, rule

if __name__ == "__main__":
    log("Read data")
    raw_dir = Path("./data/raw")
    train_df = pd.read_csv(raw_dir / "problem_train.csv", low_memory=False)
    test_df = pd.read_csv(raw_dir / "problem_test.csv", low_memory=False)
    train_labels_df = pd.read_csv(raw_dir / "problem_labels.csv")

    log("Find na cols")
    train_col_counts = train_df.describe(include="all").loc["count"]
    test_col_counts = test_df.describe(include="all").loc["count"]

    train_na_cols = train_col_counts[train_col_counts == 0].index.to_series()
    test_na_cols = test_col_counts[test_col_counts == 0].index.to_series()
    all_na_cols = pd.concat((train_na_cols, test_na_cols)).values

    log("Drop na cols")
    train_df = train_df.drop(columns=all_na_cols)
    test_df = test_df.drop(columns=all_na_cols)

    log("Save")
    prepared_dir = Path("./data/prepared")
    prepared_dir.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(prepared_dir / "train.csv", index=False)
    test_df.to_csv(prepared_dir / "test.csv", index=False)
    train_labels_df.to_csv(prepared_dir / "train_labels.csv", index=False)

    rule("Done")
