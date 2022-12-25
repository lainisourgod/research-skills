import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML

from src.config import log, rule

yaml = YAML()


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

seed = 42


def get_pipeline(
    categorical_features: np.ndarray, numerical_features: np.ndarray
) -> Pipeline:
    """
    Arguments:
        categorical_features, numerical_features: arrays with indices of columns
        that are categorical/numerical. e.g. `[0, 15, 35, 36]`
    Returns:
        sklearn.Pipeline with following steps:
        1. impute NaNs. mean for numerical and most frequent for categorical
        2. one-hot encode categorical features
        3. apply LogisticRegression with default parameters
    """
    imputer = ColumnTransformer(
        (
            (
                "impute_numerical",
                SimpleImputer(strategy="mean"),
                numerical_features,
            ),
            (
                "impute_categorical",
                SimpleImputer(strategy="most_frequent"),
                categorical_features,
            ),
        )
    )
    one_hot_encoder = ColumnTransformer(
        (
            (
                "inner",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ),
        remainder="passthrough",
    )
    model = MultiOutputClassifier(LogisticRegression(solver="liblinear"))

    return Pipeline(
        (
            ("impute", imputer),
            ("one_hot_categorical", one_hot_encoder),
            ("model", model),
        )
    )


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    * Read data
    * Impute nans
    * Convert categorical to one-hot

    Returns:
    * train features
    * train labels
    * test df
    """
    train_df = pd.read_csv("./data/prepared/train.csv", low_memory=False).drop(
        columns=["id"]
    )
    labels_df = pd.read_csv("./data/prepared/train_labels.csv", low_memory=False).drop(
        columns=["id"]
    )

    log("Labels imbalance")
    log(labels_df.apply(lambda x: pd.value_counts(x, normalize=True)).round(2).T)

    # Drop ill-defined columns
    train_counts = train_df.describe(include="all").loc["count"]
    well_defined_columns = train_counts[train_counts > 7000].index
    # Drop columns with value `1` for all records
    well_defined_columns = well_defined_columns.drop(
        ["n_0047", "n_0050", "n_0052", "n_0061", "n_0075", "n_0091"]
    )
    well_done_df = train_df[well_defined_columns]

    # We should sort columns here because ColumnTransformer will mess column order
    # So the categorical features indices will be same before and after imputing
    well_done_df = well_done_df.reindex(
        columns=sorted(well_done_df.columns, key=lambda col: well_done_df.dtypes[col])
    )

    log(f"Selected {len(well_defined_columns)} from {len(train_counts)} columns")

    test_df = pd.read_csv("./data/prepared/test.csv", low_memory=False)

    test_id = test_df["id"]
    test_df = test_df[well_defined_columns]

    # We can miss some columns from train, as well as have some new
    test_df = test_df.reindex(columns=well_done_df.columns)
    test_df["id"] = test_id

    return well_done_df, labels_df, test_df


def compute_metrics(
    labels: Iterable[str], targets: pd.DataFrame, preds: np.ndarray, probas: np.ndarray
) -> pd.DataFrame:
    """
    Compute LogLoss and ROC AUC.

    Returns pd.DataFrame with dimensions [len(labels) + 1, 2]
    """
    metrics = pd.DataFrame(index=labels)

    metrics["log_loss"] = [
        round(
            log_loss(targets.iloc[:, label_id], probas[:, label_id]),
            2,
        )
        for label_id, label in enumerate(labels)
    ]

    metrics["roc_auc"] = roc_auc_score(
        targets,
        probas,
        average=None,
    )

    metrics.loc["mean", "log_loss"] = metrics["log_loss"].mean()
    metrics.loc["mean", "roc_auc"] = metrics["roc_auc"].mean()

    return metrics.round(2)


if __name__ == "__main__":
    log_dir = Path("./logs") / f"{datetime.now():%Y-%m-%d-%H:%M}"
    log_dir.mkdir(exist_ok=True, parents=True)
    log("Current exp dir: ", log_dir)

    log("Reading data")
    data, labels, test_df = read_data()
    train_features, valid_features, train_targets, valid_targets = train_test_split(
        data, labels, random_state=seed, test_size=0.2
    )
    rule()

    # Train and validate to get metrics
    log("Training")
    pipeline = get_pipeline(
        categorical_features=np.where(data.dtypes == object)[0],
        numerical_features=np.where(data.dtypes != object)[0],
    )

    for i in range(10):
        # mask random features to simulate prod situations
        # where many columns may be NaN
        pipeline.fit(train_features, train_targets)

    valid_preds = pipeline.predict(valid_features)
    valid_probas = pipeline.predict_proba(valid_features)
    valid_probas = np.transpose([y_pred[:, 1] for y_pred in valid_probas])

    log("Computing metrics on valid set")
    metrics = compute_metrics(
        labels=labels.columns,
        targets=valid_targets,
        preds=valid_preds,
        probas=valid_probas,
    )
    log(metrics)

    log("Saving metrics")
    with open(log_dir / "metrics.yaml", "w") as fd:
        yaml.dump(metrics.to_dict(), fd)
    rule()

    log("Training prod model on all data")
    pipeline.fit(data, labels)

    log("Exporting test scores")
    test_probas = pipeline.predict_proba(test_df.drop(columns=["id"]))
    test_probas = np.transpose([y_pred[:, 1] for y_pred in test_probas])
    test_probas = pd.DataFrame(
        columns=[f"score_{label}" for label in labels.columns],
        data=test_probas,
    )
    test_probas = test_probas.round(3)
    test_probas.insert(0, "id", test_df["id"])

    test_probas.to_csv(log_dir / "problem_test_labels.csv", index=False)
    with (log_dir / "model.pickle").open("wb") as model_fd:
        pickle.dump(pipeline, model_fd)
