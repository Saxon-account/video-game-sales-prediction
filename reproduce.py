"""Reproduce the notebook's final model outside Kaggle.

Loads vgsales.csv (the Kaggle "Video Game Sales" dataset by gregorut),
trains the same pipeline the notebook ends with — imputation + one-hot
encoding into a Random Forest — and prints validation metrics.

Usage:
    python reproduce.py [path/to/vgsales.csv]

If no path is given, the dataset is downloaded from Kaggle's public
dataset endpoint (no account needed for this public dataset).
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/gregorut/videogamesales"


def get_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    local = Path("vgsales.csv")
    if local.exists():
        return local
    tmp = Path(tempfile.mkdtemp()) / "vgs.zip"
    subprocess.run(
        ["curl", "-sL", "-A", "Mozilla/5.0", DATASET_URL, "-o", str(tmp)],
        check=True,
    )
    if not zipfile.is_zipfile(tmp):
        raise RuntimeError(
            "Dataset download did not return a zip file "
            f"(got {tmp.stat().st_size} bytes from Kaggle's endpoint)."
        )
    with zipfile.ZipFile(tmp) as z:
        z.extract("vgsales.csv", local.parent)
    return local


def main() -> None:
    data = pd.read_csv(get_csv())

    y = data["Global_Sales"]
    X = data[["Name", "Genre", "Publisher", "Year"]]

    # Same split as the notebook.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=1
    )

    categorical_cols = ["Name", "Genre", "Publisher"]
    numerical_cols = ["Year"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant"), numerical_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    # The notebook's final model settings.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=50, max_depth=100, random_state=1),
            ),
        ]
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    print(f"Rows: {len(data)} total, {len(X_train)} train, {len(X_valid)} validation")
    print(f"Training MAE:   {mean_absolute_error(y_train, train_pred):.3f}")
    print(f"Validation MAE: {mean_absolute_error(y_valid, valid_pred):.3f}")
    print(f"Validation R^2: {r2_score(y_valid, valid_pred):.3f}")
    print("MAE is in millions of units (the Global_Sales unit).")


if __name__ == "__main__":
    main()
