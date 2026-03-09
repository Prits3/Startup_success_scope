import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #0A101A;
                --surface: #121a28;
                --surface-2: #172132;
                --line: rgba(164, 186, 218, 0.16);
                --text: #e8f0fb;
                --muted: #97acc8;
                --accent: #2bc6f4;
                --success: #39d49b;
                --warn: #f3b547;
                --risk: #ff6e63;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(900px 350px at 12% -12%, rgba(43,198,244,0.11), transparent 62%),
                    radial-gradient(700px 280px at 88% -12%, rgba(96,120,255,0.12), transparent 62%),
                    var(--bg);
                color: var(--text);
            }

            .block-container {
                max-width: 1140px;
                margin: 0 auto;
                padding-top: 1.1rem;
                padding-bottom: 2.2rem;
            }

            [data-testid="stSidebar"] {
                background: #0d1420;
                border-right: 1px solid var(--line);
                min-width: 220px;
                max-width: 220px;
            }

            [data-testid="stSidebarNav"] {
                display: none;
            }

            .topbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.1rem;
            }

            .brand-title {
                font-size: 1.05rem;
                font-weight: 800;
                letter-spacing: 0.01rem;
            }

            .brand-sub {
                color: var(--muted);
                font-size: 0.8rem;
            }

            .page-title {
                font-size: 1.9rem;
                font-weight: 800;
                margin: 0;
            }

            .page-sub {
                color: var(--muted);
                margin-top: 0.35rem;
                margin-bottom: 0.95rem;
            }

            .section-gap {
                margin-top: 1.1rem;
                margin-bottom: 1.1rem;
            }

            .kpi-card {
                background: rgba(18, 26, 40, 0.9);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 10px 12px;
                min-height: 80px;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.04rem;
            }

            .kpi-value {
                color: var(--accent);
                font-size: 1.2rem;
                font-weight: 700;
                margin-top: 6px;
            }

            .surface {
                background: rgba(18, 26, 40, 0.88);
                border: 1px solid var(--line);
                border-radius: 10px;
                padding: 12px;
            }

            .ai-brief {
                background: rgba(16, 37, 52, 0.68);
                border: 1px solid rgba(43, 198, 244, 0.35);
                border-left: 4px solid var(--accent);
                border-radius: 10px;
                padding: 12px;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-top: 0.9rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )
