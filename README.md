# VentureScope AI

AI-powered startup evaluation platform with predictive scoring, market intelligence, and startup comparison.

## Core Capabilities

- Success probability prediction (0-100%)
- Risk classification:
  - High Potential
  - Moderate Potential
  - High Risk
- Feature importance chart for model explainability
- Market insights by industry/country/funding bands
- Side-by-side startup comparison workflow
- Fintech dashboard UI with dark navy + teal theme

## Tech Stack

- Frontend: Streamlit
- ML model: RandomForestClassifier (scikit-learn)
- Data: CSV startup dataset (`startups_raw.csv` compatible schema)

## Project Structure

```text
venturescope-ai/
├── app.py
├── model/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── *.pkl (generated artifacts)
├── data/
│   ├── generate_data.py
│   ├── startups_raw.csv
│   └── big_startup_secsees_dataset.csv
├── pages/
│   ├── __init__.py
│   ├── startup_analysis.py
│   ├── market_insights.py
│   └── comparison.py
├── .streamlit/
│   └── config.toml
├── Procfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset Requirements

CSV columns expected:

- `category_list`
- `funding_total_usd`
- `status`
- `country_code`
- `funding_rounds`
- `founded_at`
- `first_funding_at`
- `last_funding_at`

The training pipeline auto-detects data from:

1. `data/startups_raw.csv`
2. `data/big_startup_secsees_dataset.csv`

If neither file exists, it auto-generates a synthetic dataset via `data/generate_data.py`.

## Local Setup

```bash
cd ~/Desktop/venturescope-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Model

```bash
python3 model/train.py
```

Generated artifacts:

- `model/model.pkl`
- `model/encoders.pkl`
- `model/categories.pkl`
- `model/metrics.pkl`

## Run App

```bash
streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

Open: `http://127.0.0.1:8501`

## Deployment

### Option 1: Streamlit Community Cloud

1. Push repository to GitHub
2. In Streamlit Cloud, create app from repo
3. Set main file to `app.py`
4. Ensure `requirements.txt` is present (already included)

### Option 2: Render / PaaS using Procfile

- Build command:

```bash
pip install -r requirements.txt
```

- Start command (already in `Procfile`):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Product Workflow

1. Dashboard: high-level module entry points
2. Startup Analysis: input startup profile, get AI score + risk + importance
3. Market Insights: trends by industry, country, and funding bucket
4. Compare Startups: compare saved analyses side-by-side

## Notes

- If `model/model.pkl` is missing, the app trains automatically at startup.
- Model metrics are shown in the sidebar after training.
- Keep large CSV files out of Git history; `.gitignore` already excludes data/model binaries.
