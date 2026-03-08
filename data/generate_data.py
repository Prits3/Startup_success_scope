from pathlib import Path

import numpy as np
import pandas as pd


def generate_startups_raw(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = [
        "Software",
        "FinTech",
        "Health Care",
        "E-Commerce",
        "AI",
        "EdTech",
        "Cybersecurity",
        "CleanTech",
        "Biotech",
    ]
    countries = ["USA", "GBR", "DEU", "IND", "CAN", "SGP", "FRA", "AUS", "NLD"]

    category = rng.choice(categories, n_rows)
    country = rng.choice(countries, n_rows)

    founded_base = pd.Timestamp("2008-01-01")
    founded_offsets = rng.integers(0, 15 * 365, size=n_rows)
    founded_at = founded_base + pd.to_timedelta(founded_offsets, unit="D")

    days_to_first = rng.integers(30, 1200, size=n_rows)
    duration = rng.integers(30, 2500, size=n_rows)

    first_funding_at = founded_at + pd.to_timedelta(days_to_first, unit="D")
    last_funding_at = first_funding_at + pd.to_timedelta(duration, unit="D")

    funding_rounds = rng.integers(1, 9, size=n_rows)
    funding_total_usd = np.clip(rng.lognormal(mean=14.5, sigma=1.25, size=n_rows), 5e4, 8e8)

    cat_bonus = {
        "Software": 0.08,
        "FinTech": 0.12,
        "Health Care": 0.10,
        "E-Commerce": 0.03,
        "AI": 0.14,
        "EdTech": 0.04,
        "Cybersecurity": 0.11,
        "CleanTech": 0.06,
        "Biotech": 0.09,
    }
    ctry_bonus = {
        "USA": 0.12,
        "GBR": 0.08,
        "DEU": 0.07,
        "IND": 0.05,
        "CAN": 0.06,
        "SGP": 0.06,
        "FRA": 0.05,
        "AUS": 0.04,
        "NLD": 0.05,
    }

    score = (
        0.18 * np.log1p(funding_total_usd)
        + 0.08 * funding_rounds
        - 0.0002 * days_to_first
        + 0.00018 * duration
        + np.vectorize(cat_bonus.get)(category)
        + np.vectorize(ctry_bonus.get)(country)
        - 2.3
        + rng.normal(0, 0.35, n_rows)
    )
    prob = 1 / (1 + np.exp(-score))

    status = np.where(
        prob > 0.62,
        rng.choice(["ipo", "acquired", "operating"], size=n_rows, p=[0.35, 0.5, 0.15]),
        rng.choice(["operating", "closed"], size=n_rows, p=[0.85, 0.15]),
    )

    df = pd.DataFrame(
        {
            "permalink": [f"/organization/startup-{i}" for i in range(n_rows)],
            "name": [f"Startup {i}" for i in range(n_rows)],
            "homepage_url": [f"https://startup{i}.example.com" for i in range(n_rows)],
            "category_list": category,
            "funding_total_usd": funding_total_usd.astype(int),
            "status": status,
            "country_code": country,
            "state_code": "NA",
            "region": "Global",
            "city": "Unknown",
            "funding_rounds": funding_rounds,
            "founded_at": founded_at.astype(str),
            "first_funding_at": first_funding_at.astype(str),
            "last_funding_at": last_funding_at.astype(str),
        }
    )
    return df


def main() -> Path:
    out = Path(__file__).resolve().parent / "startups_raw.csv"
    df = generate_startups_raw()
    df.to_csv(out, index=False)
    print(f"Generated synthetic dataset: {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    main()
