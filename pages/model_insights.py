import pandas as pd
import plotly.express as px
import streamlit as st

from model.predict import get_feature_importance, load_metrics
from ui import render_kpi


def render(df: pd.DataFrame) -> None:
    st.markdown("<p class='page-title'>Model</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-sub'>Model transparency for trust: what drives scores, and where caution is needed.</p>",
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Model", "Random Forest")
    with k2:
        render_kpi("Dataset Size", f"{len(df):,}")
    with k3:
        render_kpi("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    with k4:
        render_kpi("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    fi = get_feature_importance(top_n=12).sort_values("importance")
    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#1f2e46", "#2bc6f4"],
        title="Feature Importance",
    )
    fig.update_layout(
        paper_bgcolor="#0A101A",
        plot_bgcolor="#0A101A",
        font_color="#e8f0fb",
        coloraxis_showscale=False,
        xaxis_title="Relative Influence",
        yaxis_title="",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("<div class='ai-brief'><b>Trust Note</b><br/>"
                "Use outputs for screening, not final decisions. Combine score with founder quality, market timing, and competitive context."
                "</div>", unsafe_allow_html=True)
