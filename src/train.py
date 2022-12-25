import pickle
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML

from src.config import log

yaml = YAML()


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

seed = 42


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
    train_labels_df = pd.read_csv(
        "./data/prepared/train_labels.csv", low_memory=False
    ).drop(columns=["id"])

    log("Labels imbalance")
    log(train_labels_df.apply(lambda x: pd.value_counts(x, normalize=True)).round(2))

    train_counts = train_df.describe(include="all").loc["count"]
    well_defined_columns = train_counts[train_counts > 7000].index
    well_done_df = train_df[well_defined_columns]

    log(f"Selected {len(well_defined_columns)} from {len(train_counts)} columns")

    # Convert categorical. `nan` converts to [0, 0, ..., 0]
    well_done_df = pd.get_dummies(well_done_df)

    # TODO:!!!
    # simple imputer should be learned only on train, not valid (️️️️️️🤦‍♂️)
    # maybe put it into pipeline?

    # Impute nans for numerical features
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(well_done_df)
    well_done_df = pd.DataFrame(
        columns=well_done_df.columns,
        data=imputer.transform(well_done_df),
    )

    test_df = pd.read_csv("./data/prepared/test.csv", low_memory=False)

    test_id = test_df["id"]
    test_df = test_df[well_defined_columns]

    test_df = pd.get_dummies(test_df)

    # We can miss some columns from train, as well as have some new
    test_df = test_df.reindex(columns=well_done_df.columns, fill_value=0)

    test_df = pd.DataFrame(
        columns=test_df.columns,
        data=imputer.transform(test_df),
    )
    test_df["id"] = test_id

    return well_done_df, train_labels_df, test_df


def compute_metrics(
    labels: Iterable[str], targets: pd.DataFrame, preds: np.ndarray, probas: np.ndarray
) -> pd.DataFrame:
    """
    Compute LogLoss and ROC AUC.

    Returns pd.DataFrame with dimensions [len(labels), len(metrics) (2)]
    """
    metrics = pd.DataFrame(index=labels)

    metrics["log_loss"] = [
        round(
            log_loss(targets.iloc[:, label_id], probas[:, label_id]),
            2,
        )
        for label_id, label in enumerate(train_labels_df.columns)
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
    well_done_df, train_labels_df, test_df = read_data()

    # Train and validate to get metrics
    log("Training")
    train_features, valid_features, train_targets, valid_targets = train_test_split(
        well_done_df, train_labels_df, random_state=seed, test_size=0.2
    )

    # We need MultiOutputClassifier becase CalibratedClassifierCV only outputs one label
    clf = MultiOutputClassifier(
        LogisticRegression(solver="liblinear", random_state=seed)
    ).fit(well_done_df, train_labels_df)

    valid_preds = clf.predict(valid_features)
    valid_probas = clf.predict_proba(valid_features)
    valid_probas = np.transpose([y_pred[:, 1] for y_pred in valid_probas])

    log("Computing metrics on valid set")
    metrics = compute_metrics(
        labels=train_labels_df.columns,
        targets=valid_targets,
        preds=valid_preds,
        probas=valid_probas,
    )
    log(metrics)

    log("Saving metrics")
    with open(log_dir / "metrics.yaml", "w") as fd:
        yaml.dump(metrics.to_dict(), fd)

    log("Training prod model on all data")
    # Use all data to train prod model
    clf = MultiOutputClassifier(
        LogisticRegression(solver="liblinear", random_state=seed)
    ).fit(well_done_df, train_labels_df)

    log("Exporting test scores")
    test_probas = clf.predict_proba(test_df.drop(columns=["id"]))
    test_probas = np.transpose([y_pred[:, 1] for y_pred in test_probas])
    test_probas = pd.DataFrame(
        columns=[f"score_{label}" for label in train_labels_df.columns],
        data=test_probas,
    )
    test_probas = test_probas.round(3)
    test_probas.insert(0, "id", test_df["id"])

    test_probas.to_csv(log_dir / "problem_test_labels.csv", index=False)
    with (log_dir / "model.pickle").open("wb") as model_fd:
        pickle.dump(clf, model_fd)
