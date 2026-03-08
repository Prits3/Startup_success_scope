import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.markdown("## ⚖️ Startup Comparison")
    if 'compared_startups' not in st.session_state or not st.session_state.compared_startups:
        st.info("👈 Go to **Startup Analysis**, analyze a startup, then click **Add to Comparison**.")
        return
    df = pd.DataFrame(st.session_state.compared_startups)
    st.dataframe(df, use_container_width=True)
    df['Label'] = df['Industry'] + ' (' + df['Country'] + ')'
    fig = px.bar(df, x='Label', y='Probability (%)', color='Probability (%)', color_continuous_scale=['#FF4B4B','#F5A623','#00C897'], text='Probability (%)')
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(paper_bgcolor='#0A1628', plot_bgcolor='#0A1628', font_color='white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    if st.button("🗑️ Clear All"):
        st.session_state.compared_startups = []
        st.rerun()
