# VentureScope AI

VentureScope AI is an AI-powered startup evaluation platform built with Streamlit and a RandomForest model.

## Features

- Startup success probability prediction (0-100%)
- Risk classification:
  - High Potential
  - Moderate Potential
  - High Risk
- Feature importance visualization
- Market insights dashboard with industry/location trends
- Side-by-side startup comparison tool
- Professional fintech style with dark navy and teal theme

## Project Structure

```text
venturescope-ai/
├── app.py
├── model/
│   ├── train.py
│   └── predict.py
├── data/
│   └── generate_data.py
├── pages/
│   ├── startup_analysis.py
│   ├── market_insights.py
│   └── comparison.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

1. Generate synthetic data:

```bash
python data/generate_data.py
```

2. Train the model:

```bash
python model/train.py
```

3. Launch Streamlit app:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

4. Open in browser:

```text
http://localhost:8501
```

## Input Features

- `funding_amount`
- `num_funding_rounds`
- `industry_sector`
- `location`
- `num_milestones`
- `team_size`
- `years_active`

## Notes

- Model artifacts are saved to `model/startup_success_model.joblib` and `model/training_metrics.joblib`.
- If model files are missing, run training before using prediction pages.
