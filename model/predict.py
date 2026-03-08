from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"


@lru_cache(maxsize=1)
def _load_model_bundle():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def predict_startup(funding_usd, funding_rounds, category, country, days_to_funding, funding_duration):
    model, encoders = _load_model_bundle()

    cats = set(encoders["category"].classes_)
    ctrys = set(encoders["country"].classes_)

    cat_enc = encoders["category"].transform([category])[0] if category in cats else 0
    ctry_enc = encoders["country"].transform([country])[0] if country in ctrys else 0

    X = np.array(
        [[funding_usd, funding_rounds, cat_enc, ctry_enc, days_to_funding, funding_duration]],
        dtype=float,
    )
    prob = float(model.predict_proba(X)[0][1])
    pct = round(prob * 100, 1)

    if pct >= 40:
        risk, signal, color = "High Potential", "Strong Investment Signal", "#00C897"
    elif pct >= 20:
        risk, signal, color = "Moderate Potential", "Promising Early Stage", "#F5A623"
    else:
        risk, signal, color = "High Risk", "Proceed with Caution", "#FF4B4B"

    names = [
        "Funding Amount",
        "Funding Rounds",
        "Industry",
        "Country",
        "Days to Funding",
        "Funding Duration",
    ]
    importance = dict(zip(names, model.feature_importances_))

    return {
        "probability": pct,
        "risk_level": risk,
        "signal": signal,
        "color": color,
        "feature_importance": importance,
    }
