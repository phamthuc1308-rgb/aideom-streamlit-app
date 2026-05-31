from __future__ import annotations
import streamlit as st

TEAL = '#2a9d8f'
CORAL = '#e76f51'
NAVY = '#1f3a5f'
GOLD = '#f4a261'
LIGHT_GREEN = '#f0f7f4'
LIGHT_YELLOW = '#fff8e7'


def inject_css() -> None:
    st.markdown(
        f'''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: Inter, Segoe UI, system-ui, sans-serif; }}
        .stApp {{ background: linear-gradient(180deg, #f7fbff 0%, #eef7f5 100%); color: #0f172a; }}
        section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0f2b3d 0%, #1a4a6f 100%); }}
        section[data-testid="stSidebar"] * {{ color: #ffffff !important; }}
        section[data-testid="stSidebar"] .stButton button {{
            background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); border-radius: 12px;
        }}
        .aideom-banner {{
            border-radius: 16px; padding: 1.2rem 1.4rem; margin: .4rem 0 1rem 0;
            background: linear-gradient(135deg, {NAVY}, {TEAL}); color: white;
            box-shadow: 0 12px 30px rgba(31,58,95,.18);
        }}
        .aideom-banner h1 {{ font-size: 1.6rem; margin: 0 0 .35rem 0; font-weight: 800; }}
        .aideom-banner p {{ margin: 0; opacity: .92; font-size: .98rem; }}
        .aideom-card {{
            background: rgba(255,255,255,.96); border: 1px solid #dbe7ef; border-radius: 12px;
            padding: 1.5rem; margin: .75rem 0 1rem 0; box-shadow: 0 8px 24px rgba(15,43,61,.07);
        }}
        .analysis-box {{
            background: {LIGHT_GREEN}; border-left: 5px solid {TEAL}; border-radius: 8px;
            padding: 1rem 1.1rem; margin: 1rem 0; box-shadow: 0 6px 16px rgba(42,157,143,.08);
        }}
        .analysis-box.yellow {{ background: {LIGHT_YELLOW}; border-left-color: {GOLD}; }}
        .analysis-box h4 {{ margin: 0 0 .55rem 0; color: #17324d; }}
        .caption {{ color: #52657a; font-size: .88rem; margin-top: -.35rem; margin-bottom: .7rem; }}
        .metric-card {{
            background: white; border: 1px solid #dbe7ef; border-radius: 12px; padding: 1rem;
            box-shadow: 0 6px 18px rgba(15,43,61,.06);
        }}
        div[data-testid="stMetric"] {{ background: white; border: 1px solid #dbe7ef; padding: 1rem; border-radius: 12px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
        .stTabs [data-baseweb="tab"] {{ border-radius: 999px; background: #e8f3f1; padding: .4rem .8rem; }}
        </style>
        ''',
        unsafe_allow_html=True,
    )


def banner(title: str, desc: str, icon: str = '📊') -> None:
    st.markdown(
        f'<div class="aideom-banner"><h1>{icon} {title}</h1><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )


def card_start() -> None:
    st.markdown('<div class="aideom-card">', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def analysis_box(main: str, explain: str, policy: str, warning: str | None = None, icon: str = '🔍', yellow: bool = False) -> None:
    cls = 'analysis-box yellow' if yellow else 'analysis-box'
    warn_html = f'<li><b>⚠️ Cảnh báo:</b> {warning}</li>' if warning else ''
    st.markdown(
        f'''
        <div class="{cls}">
            <h4>{icon} Tác nhân phân tích kết quả</h4>
            <ul>
                <li><b>Kết quả chính:</b> {main}</li>
                <li><b>Giải thích:</b> {explain}</li>
                <li><b>Hàm ý chính sách:</b> {policy}</li>
                {warn_html}
            </ul>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def caption(text: str) -> None:
    st.markdown(f'<div class="caption">{text}</div>', unsafe_allow_html=True)
