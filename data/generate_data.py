import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INDUSTRIES = [
    "FinTech",
    "HealthTech",
    "EdTech",
    "SaaS",
    "AI/ML",
    "E-commerce",
    "CleanTech",
    "Cybersecurity",
]

LOCATIONS = [
    "San Francisco",
    "New York",
    "Austin",
    "Boston",
    "Seattle",
    "London",
    "Berlin",
    "Singapore",
    "Toronto",
    "Bangalore",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_startup_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    funding_amount = np.round(rng.lognormal(mean=2.9, sigma=1.0, size=n_samples), 2)
    funding_amount = np.clip(funding_amount, 0.25, 600.0)

    num_funding_rounds = rng.integers(1, 8, size=n_samples)
    industry_sector = rng.choice(INDUSTRIES, size=n_samples)
    location = rng.choice(LOCATIONS, size=n_samples)
    num_milestones = rng.integers(0, 18, size=n_samples)
    team_size = rng.integers(2, 250, size=n_samples)
    years_active = np.round(rng.uniform(0.3, 12.0, size=n_samples), 1)

    industry_bonus = {
        "FinTech": 0.20,
        "HealthTech": 0.18,
        "EdTech": 0.07,
        "SaaS": 0.15,
        "AI/ML": 0.23,
        "E-commerce": 0.05,
        "CleanTech": 0.11,
        "Cybersecurity": 0.17,
    }

    location_bonus = {
        "San Francisco": 0.22,
        "New York": 0.18,
        "Austin": 0.12,
        "Boston": 0.14,
        "Seattle": 0.15,
        "London": 0.16,
        "Berlin": 0.10,
        "Singapore": 0.12,
        "Toronto": 0.09,
        "Bangalore": 0.08,
    }

    latent_score = (
        0.028 * np.log1p(funding_amount)
        + 0.085 * num_funding_rounds
        + 0.11 * np.log1p(num_milestones)
        + 0.07 * np.log1p(team_size)
        + 0.06 * years_active
        + np.vectorize(industry_bonus.get)(industry_sector)
        + np.vectorize(location_bonus.get)(location)
        - 1.05
        + rng.normal(0, 0.18, size=n_samples)
    )

    success_probability = np.clip(_sigmoid(latent_score), 0.02, 0.98)
    success_outcome = rng.binomial(1, success_probability)

    return pd.DataFrame(
        {
            "funding_amount": funding_amount,
            "num_funding_rounds": num_funding_rounds,
            "industry_sector": industry_sector,
            "location": location,
            "num_milestones": num_milestones,
            "team_size": team_size,
            "years_active": years_active,
            "success_probability": np.round(success_probability * 100, 2),
            "success_outcome": success_outcome,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic startup dataset.")
    parser.add_argument("--samples", type=int, default=3000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / "startup_data.csv"
    df = generate_synthetic_startup_data(n_samples=args.samples, random_state=args.seed)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
