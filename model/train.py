from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"


def resolve_data_path() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "startups_raw.csv",
        PROJECT_ROOT / "data" / "big_startup_secsees_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    # Fallback: generate synthetic data if no CSV is present.
    from data.generate_data import main as generate_main

    return generate_main()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["success"] = df["status"].apply(lambda x: 1 if str(x).lower() in ["acquired", "ipo"] else 0)
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce").fillna(0)
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce").fillna(1)

    df["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")
    df["first_funding_at"] = pd.to_datetime(df["first_funding_at"], errors="coerce")
    df["last_funding_at"] = pd.to_datetime(df["last_funding_at"], errors="coerce")

    df["days_to_first_funding"] = (df["first_funding_at"] - df["founded_at"]).dt.days
    df["days_to_first_funding"] = df["days_to_first_funding"].fillna(365).clip(lower=0)

    df["funding_duration_days"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days
    df["funding_duration_days"] = df["funding_duration_days"].fillna(0).clip(lower=0)

    df["primary_category"] = df["category_list"].fillna("Unknown").astype(str).apply(lambda x: x.split("|")[0])
    df["country_code"] = df["country_code"].fillna("Unknown").astype(str)
    return df


def train_model() -> dict:
    data_path = resolve_data_path()
    df = pd.read_csv(data_path)
    df = _prepare_dataframe(df)

    le_cat = LabelEncoder()
    le_country = LabelEncoder()
    df["category_encoded"] = le_cat.fit_transform(df["primary_category"])
    df["country_encoded"] = le_country.fit_transform(df["country_code"])

    features = [
        "funding_total_usd",
        "funding_rounds",
        "category_encoded",
        "country_encoded",
        "days_to_first_funding",
        "funding_duration_days",
    ]

    X = df[features]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=14,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "rows": int(len(df)),
        "data_path": str(data_path),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "encoders.pkl", "wb") as f:
        pickle.dump({"category": le_cat, "country": le_country}, f)
    with open(MODEL_DIR / "categories.pkl", "wb") as f:
        pickle.dump({"categories": list(le_cat.classes_), "countries": list(le_country.classes_)}, f)
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(f"Trained on {metrics['rows']:,} rows from {Path(metrics['data_path']).name}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    return metrics


if __name__ == "__main__":
    train_model()
