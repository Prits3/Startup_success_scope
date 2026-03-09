import pandas as pd
import plotly.express as px
import streamlit as st

from ui import render_kpi


def _market_brief(df: pd.DataFrame) -> str:
    sector_rank = df.groupby("industry_sector")["success_probability"].mean().sort_values(ascending=False)
    top_two = list(sector_rank.index[:2]) if len(sector_rank) >= 2 else list(sector_rank.index[:1])
    if len(top_two) == 1:
        top_two.append(top_two[0])
    best_region = df.groupby("location")["success_probability"].mean().sort_values(ascending=False).index[0]
    return (
        f"{top_two[0]} and {top_two[1]} currently show the strongest momentum. "
        f"{best_region} leads regional startup quality signals in this dataset."
    )


def render(df: pd.DataFrame) -> None:
    st.markdown("<p class='page-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>AI venture intelligence for startup screening and market analysis.</p>",
        unsafe_allow_html=True,
    )

    hero_l, hero_r = st.columns([1.35, 1])
    with hero_l:
        c1, c2 = st.columns(2)
        if c1.button("Analyze Startup", key="overview_cta_analyze", width='stretch', type="primary"):
            st.info("Use the left navigation and open Analyze.")
        if c2.button("Compare Startups", key="overview_cta_compare", width='stretch'):
            st.info("Use the left navigation and open Compare.")
    with hero_r:
        st.markdown("<div class='ai-brief'><b>AI Market Brief</b><br/>" + _market_brief(df) + "</div>", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi("Startups Analyzed", f"{len(df):,}")
    with k2:
        render_kpi("Avg Success Rate", f"{df['success_outcome'].mean() * 100:.1f}%")
    with k3:
        top_sector = df.groupby("industry_sector")["success_probability"].mean().sort_values(ascending=False).index[0]
        render_kpi("Top Sector", top_sector)

    sector = (
        df.groupby("industry_sector", as_index=False)
        .agg(avg_prob=("success_probability", "mean"), avg_funding=("funding_amount", "mean"))
        .sort_values("avg_prob", ascending=False)
    )

    fig = px.scatter(
        sector,
        x="avg_funding",
        y="avg_prob",
        size="avg_prob",
        color="industry_sector",
        title="Market Momentum",
    )
    fig.update_layout(
        paper_bgcolor="#0A101A",
        plot_bgcolor="#0A101A",
        font_color="#e8f0fb",
        xaxis_title="Average Funding (M$)",
        yaxis_title="Average Success Probability (%)",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Top Performing Sectors")
    sector_table = (
        df.groupby("industry_sector", as_index=False)
        .agg(success=("success_probability", "mean"), avg_funding=("funding_amount", "mean"))
        .sort_values("success", ascending=False)
        .head(8)
    )
    st.dataframe(sector_table.round(1), width='stretch', hide_index=True)
