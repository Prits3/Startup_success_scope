from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _data_path() -> Path:
    for p in [
        PROJECT_ROOT / "data" / "startups_raw.csv",
        PROJECT_ROOT / "data" / "big_startup_secsees_dataset.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("Startup dataset not found in data/")


def show():
    st.markdown("## Market Insights")

    df = pd.read_csv(_data_path())
    df["success"] = df["status"].apply(lambda x: 1 if str(x).lower() in ["acquired", "ipo"] else 0)
    df["primary_category"] = df["category_list"].fillna("Unknown").astype(str).apply(lambda x: x.split("|")[0])
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce").fillna(0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Startups", f"{len(df):,}")
    c2.metric("Success Rate", f"{df['success'].mean():.1%}")
    c3.metric("Avg Funding", f"${df['funding_total_usd'].mean():,.0f}")
    c4.metric("Countries", f"{df['country_code'].nunique()}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Industries by Success Rate")
        ind = df.groupby("primary_category")["success"].agg(["mean", "count"]).reset_index()
        ind = ind[ind["count"] > 50].sort_values("mean", ascending=True).tail(15)
        fig = px.bar(
            ind,
            x="mean",
            y="primary_category",
            orientation="h",
            color="mean",
            color_continuous_scale=["#0A1628", "#00B4D8"],
        )
        fig.update_layout(
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font_color="white",
            showlegend=False,
            xaxis_title="Success Rate",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top Countries by Success Rate")
        ctry = df.groupby("country_code")["success"].agg(["mean", "count"]).reset_index()
        ctry = ctry[ctry["count"] > 100].sort_values("mean", ascending=False).head(15)
        fig2 = px.bar(
            ctry,
            x="country_code",
            y="mean",
            color="mean",
            color_continuous_scale=["#0A1628", "#00C897"],
        )
        fig2.update_layout(
            paper_bgcolor="#0A1628",
            plot_bgcolor="#0A1628",
            font_color="white",
            showlegend=False,
            yaxis_title="Success Rate",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Funding vs Success")
    df["funding_bucket"] = pd.cut(
        df["funding_total_usd"],
        bins=[0, 100_000, 1_000_000, 10_000_000, 100_000_000, 500_000_000, float("inf")],
        labels=["<100K", "100K-1M", "1M-10M", "10M-100M", "100M-500M", "500M+"],
    )
    fb = df.groupby("funding_bucket", observed=True)["success"].mean().reset_index()
    fig3 = px.bar(
        fb,
        x="funding_bucket",
        y="success",
        color="success",
        color_continuous_scale=["#0A1628", "#00B4D8"],
    )
    fig3.update_layout(
        paper_bgcolor="#0A1628",
        plot_bgcolor="#0A1628",
        font_color="white",
        showlegend=False,
        xaxis_title="Funding Range",
        yaxis_title="Success Rate",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
