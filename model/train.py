import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "funding_amount",
    "num_funding_rounds",
    "industry_sector",
    "location",
    "num_milestones",
    "team_size",
    "years_active",
]

TARGET_COLUMN = "success_outcome"


def train_model(data_path: Path) -> tuple[Pipeline, dict]:
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    categorical_features = ["industry_sector", "location"]
    numeric_features = [col for col in FEATURE_COLUMNS if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return pipeline, metrics


def save_artifacts(pipeline: Pipeline, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / "startup_success_model.joblib"
    metrics_file = output_dir / "training_metrics.joblib"

    joblib.dump(pipeline, model_file)
    joblib.dump(metrics, metrics_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train startup success prediction model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "startup_data.csv"),
        help="Path to CSV training data.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(__file__).resolve().parent

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run data/generate_data.py first."
        )

    pipeline, metrics = train_model(data_path)
    save_artifacts(pipeline, metrics, output_dir)

    print("Model trained and saved successfully.")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"Train rows: {metrics['train_rows']}, Test rows: {metrics['test_rows']}")


if __name__ == "__main__":
    main()
