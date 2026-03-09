import pandas as pd
import plotly.express as px
import streamlit as st


LOCATION_TO_COUNTRY = {
    "San Francisco": "United States",
    "New York": "United States",
    "Austin": "United States",
    "Boston": "United States",
    "Seattle": "United States",
    "London": "United Kingdom",
    "Berlin": "Germany",
    "Singapore": "Singapore",
    "Toronto": "Canada",
    "Bangalore": "India",
}


def _market_brief(df: pd.DataFrame) -> str:
    sectors = (
        df.groupby("industry_sector", as_index=False)
        .agg(success=("success_probability", "mean"), funding=("funding_amount", "mean"))
        .sort_values("success", ascending=False)
    )
    s1 = sectors.iloc[0]["industry_sector"]
    s2 = sectors.iloc[1]["industry_sector"]
    spread = sectors["success"].max() - sectors["success"].min()
    return (
        f"{s1} and {s2} currently lead the signal stack. "
        f"Sector outcome spread is {spread:.1f} points, indicating meaningful dispersion in venture quality."
    )


def render(df: pd.DataFrame) -> None:
    st.subheader("Signals")
    st.caption("Ecosystem-level market signals for strategic venture decisions.")

    st.markdown("<div class='ai-note'><b>AI Market Brief</b><br/>" + _market_brief(df) + "</div>", unsafe_allow_html=True)

    sector = (
        df.groupby("industry_sector", as_index=False)
        .agg(success_rate=("success_outcome", "mean"), avg_funding=("funding_amount", "mean"))
    )
    sector["success_rate"] = sector["success_rate"] * 100

    fig1 = px.density_heatmap(
        df,
        x="industry_sector",
        y="location",
        z="success_probability",
        histfunc="avg",
        color_continuous_scale=["#1b2b46", "#27c9f5"],
        title="Sector-Region Heatmap (Avg Success Probability)",
    )
    fig1.update_layout(
        paper_bgcolor="#070d17",
        plot_bgcolor="#070d17",
        font_color="#e7eef9",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig1, width='stretch')

    row1, row2 = st.columns(2)
    with row1:
        fig2 = px.bar(
            sector.sort_values("success_rate", ascending=False),
            x="industry_sector",
            y="success_rate",
            color="success_rate",
            title="Strongest Sectors",
            color_continuous_scale=["#1b2b46", "#27c9f5"],
        )
        fig2.update_layout(
            paper_bgcolor="#070d17",
            plot_bgcolor="#070d17",
            font_color="#e7eef9",
            yaxis_title="Success Rate (%)",
            xaxis_title="Industry",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, width='stretch')

    with row2:
        stage = df.copy()
        stage["stage"] = pd.cut(
            stage["num_funding_rounds"],
            bins=[0, 2, 4, 6, 10],
            labels=["Early", "Growth", "Late", "Scale"],
        )
        stage_summary = stage.groupby("stage", as_index=False).agg(
            avg_success=("success_probability", "mean"),
            avg_funding=("funding_amount", "mean"),
        )
        fig3 = px.line(
            stage_summary,
            x="stage",
            y="avg_success",
            markers=True,
            title="Funding Stage Analysis",
            color_discrete_sequence=["#7c8cff"],
        )
        fig3.update_layout(
            paper_bgcolor="#070d17",
            plot_bgcolor="#070d17",
            font_color="#e7eef9",
            yaxis_title="Avg Success Probability (%)",
            xaxis_title="Stage",
        )
        st.plotly_chart(fig3, width='stretch')

    geo_df = df.copy()
    geo_df["country"] = geo_df["location"].map(LOCATION_TO_COUNTRY)
    region_success = geo_df.groupby("country", as_index=False).agg(
        success=("success_probability", "mean"),
        count=("location", "count"),
    )

    fig4 = px.choropleth(
        region_success,
        locations="country",
        locationmode="country names",
        color="success",
        hover_data=["count"],
        color_continuous_scale=["#1b2b46", "#27c9f5"],
        title="Geography Clusters and Performance",
    )
    fig4.update_layout(
        paper_bgcolor="#070d17",
        plot_bgcolor="#070d17",
        font_color="#e7eef9",
        geo=dict(bgcolor="#070d17", showcoastlines=True, showframe=False),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig4, width='stretch')
