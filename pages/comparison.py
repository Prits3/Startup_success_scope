import pandas as pd
import plotly.express as px
import streamlit as st

from model.predict import predict_startup


def _input_col(label: str, industries: list[str], locations: list[str], default_funding: float) -> dict:
    st.markdown(f"**{label}**")
    industry = st.selectbox(f"{label} Industry", industries, key=f"{label}_industry")
    location = st.selectbox(f"{label} Location", locations, key=f"{label}_location")
    funding = st.slider(f"{label} Funding (M$)", 0.5, 250.0, default_funding, 0.5, key=f"{label}_funding")
    rounds = st.slider(f"{label} Funding Rounds", 1, 10, 3, key=f"{label}_rounds")
    milestones = st.slider(f"{label} Milestones", 0, 20, 7, key=f"{label}_milestones")
    team = st.slider(f"{label} Team Size", 2, 500, 35, key=f"{label}_team")
    years = st.slider(f"{label} Years Active", 0.2, 15.0, 3.0, key=f"{label}_years")
    return {
        "funding_amount": float(funding),
        "num_funding_rounds": int(rounds),
        "industry_sector": industry,
        "location": location,
        "num_milestones": int(milestones),
        "team_size": int(team),
        "years_active": float(years),
    }


def _summary(table: pd.DataFrame) -> str:
    best = table.sort_values("Success Probability", ascending=False).iloc[0]["Startup"]
    return (
        f"{best} leads on predicted upside. Prioritize it for investment committee review, "
        "while using the second profile as a balanced-risk alternative."
    )


def render(df: pd.DataFrame) -> None:
    st.markdown("<p class='page-title'>Compare</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>Compare two startup profiles to decide which opportunity looks stronger.</p>",
        unsafe_allow_html=True,
    )

    industries = sorted(df["industry_sector"].unique().tolist())
    locations = sorted(df["location"].unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        a = _input_col("Startup A", industries, locations, 10.0)
    with c2:
        b = _input_col("Startup B", industries, locations, 15.0)

    if not st.button("Compare Startups", type="primary", width='stretch', key="compare_submit"):
        return

    rows = []
    for name, payload in {"Startup A": a, "Startup B": b}.items():
        try:
            pred = predict_startup(payload)
        except Exception as exc:
            st.error(
                "Comparison failed because prediction failed. Run `python model/train.py` "
                "to regenerate model artifacts, then retry."
            )
            st.code(str(exc))
            return
        rows.append(
            {
                "Startup": name,
                "Success Probability": pred["success_probability"],
                "Risk Level": pred["risk_level"],
                "Funding": payload["funding_amount"],
                "Rounds": payload["num_funding_rounds"],
            }
        )

    table = pd.DataFrame(rows)

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Recommended", table.sort_values("Success Probability", ascending=False).iloc[0]["Startup"])
    with r2:
        spread = table["Success Probability"].max() - table["Success Probability"].min()
        st.metric("Probability Spread", f"{spread:.1f} pts")

    fig = px.bar(
        table,
        x="Startup",
        y="Success Probability",
        color="Success Probability",
        text_auto=".1f",
        color_continuous_scale=["#ff6e63", "#f3b547", "#39d49b"],
        title="Comparison Score",
    )
    fig.update_layout(
        paper_bgcolor="#0A101A",
        plot_bgcolor="#0A101A",
        font_color="#e8f0fb",
        yaxis_range=[0, 100],
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Comparison Result")
    st.dataframe(table, width='stretch', hide_index=True)

    st.markdown("<div class='ai-brief'><b>AI Recommendation</b><br/>" + _summary(table) + "</div>", unsafe_allow_html=True)
