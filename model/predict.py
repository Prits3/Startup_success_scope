from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "startup_success_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "training_metrics.joblib"

FEATURE_COLUMNS = [
    "funding_amount",
    "num_funding_rounds",
    "industry_sector",
    "location",
    "num_milestones",
    "team_size",
    "years_active",
]


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file missing. Run `python model/train.py` after generating data."
        )
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return joblib.load(METRICS_PATH)


def _risk_label(probability_pct: float) -> str:
    if probability_pct >= 70:
        return "High Potential"
    if probability_pct >= 45:
        return "Moderate Potential"
    return "High Risk"


def predict_startup(startup_features: dict) -> dict:
    model = load_model()
    input_df = pd.DataFrame([startup_features], columns=FEATURE_COLUMNS)

    probability = float(model.predict_proba(input_df)[0][1] * 100)
    label = _risk_label(probability)

    return {
        "success_probability": round(probability, 2),
        "risk_level": label,
    }


def get_feature_importance(top_n: int = 12) -> pd.DataFrame:
    model = load_model()

    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    # Clean transformer prefixes for display.
    fi_df["feature"] = (
        fi_df["feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("industry_sector_", "industry:", regex=False)
        .str.replace("location_", "location:", regex=False)
    )

    return fi_df.head(top_n).reset_index(drop=True)
