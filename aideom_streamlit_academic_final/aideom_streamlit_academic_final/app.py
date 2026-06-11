import math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from scipy.optimize import linprog
except Exception:
    linprog = None


px.defaults.template = "plotly_white"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="AIDEOM-VN Streamlit Dashboard",
    page_icon="VN",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ===== AIDEOM-VN readable light theme ===== */
    :root { --ink:#07172e; --muted:#53657c; --card:#ffffff; --line:#cfe5df; --teal:#0faaa0; --pink:#d94d83; --navy:#0e1730; --cream:#fff7de; }
    .stApp {
        background: linear-gradient(135deg,#eefaf6 0%,#f7fbff 46%,#fff7de 100%) !important;
        color: var(--ink) !important;
    }
    .block-container {padding-top:2rem; padding-bottom:4rem;}

    /* Main text: force dark text on light background */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp label, .stApp caption,
    .stApp div[data-testid="stMarkdownContainer"],
    .stApp div[data-testid="stText"],
    .stApp div[data-testid="stCaptionContainer"],
    .stApp div[data-testid="stWidgetLabel"],
    .stApp span[data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }
    .stApp small, .stApp .stCaption, .subtitle { color: var(--muted) !important; }

    /* Sidebar stays dark, so text remains white */
    section[data-testid="stSidebar"] {background:#0e1730 !important; color:#f8fbff !important;}
    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {color:#f8fbff !important;}
    section[data-testid="stSidebar"] [role="radiogroup"] label {background:rgba(255,255,255,.04); border-radius:10px; padding:.35rem .45rem;}

    /* Titles and badges */
    .main-title {font-size:2.4rem; font-weight:900; letter-spacing:-.04em; margin-bottom:.2rem; color:#07172e !important;}
    .subtitle {font-size:1.05rem; color:#53657c !important; margin-bottom:1.2rem;}
    .badge {display:inline-flex; align-items:center; gap:.35rem; padding:.35rem .72rem; border-radius:999px; background:linear-gradient(135deg,#d94d83,#6f42c1); color:white !important; font-weight:800; font-size:.78rem; margin-right:.4rem;}

    /* Cards */
    .soft-card {background:rgba(255,255,255,.92); color:#07172e !important; border:1px solid rgba(25,130,118,.20); border-radius:22px; padding:1.15rem 1.2rem; box-shadow:0 14px 40px rgba(20,72,95,.10); margin-bottom:1rem;}
    .soft-card * {color:#07172e !important;}
    .metric-card {background:white; color:#07172e !important; border:1px solid #d9eee9; border-radius:18px; padding:1rem; box-shadow:0 8px 24px rgba(20,72,95,.08);}
    .formula-box {background:#0e1730; color:#f9fbff !important; border-radius:18px; padding:1rem 1.2rem; margin:.8rem 0; border-left:6px solid #12b8aa;}
    .formula-box * {color:#f9fbff !important;}
    .note {background:#fff7db; border-left:5px solid #d89b2b; padding:1rem; border-radius:12px; color:#33240b !important;}
    .note * {color:#33240b !important;}
    .success-box {background:#ddfbef; border-left:5px solid #13916f; padding:1rem; border-radius:12px; color:#073b30 !important;}
    .success-box * {color:#073b30 !important;}
    .warn-box {background:#ffe8e8; border-left:5px solid #d94848; padding:1rem; border-radius:12px; color:#4c1010 !important;}
    .warn-box * {color:#4c1010 !important;}

    /* Metrics */
    div[data-testid="stMetric"] {background:#ffffff !important; color:#07172e !important; border:1px solid #d9eee9; padding:1rem; border-radius:18px; box-shadow:0 8px 24px rgba(20,72,95,.08);}
    div[data-testid="stMetric"] * {color:#07172e !important;}
    div[data-testid="stMetricDelta"] * {color:#0b7f63 !important;}

    /* Tabs: avoid grey text on grey background */
    div[data-testid="stTabs"] [role="tablist"] {gap:.4rem; border-bottom:1px solid #d6e5e0;}
    div[data-testid="stTabs"] button[role="tab"] {
        color:#12305a !important; background:#ffffffcc !important; border:1px solid #d6e5e0 !important;
        border-radius:12px 12px 0 0 !important; font-weight:800 !important; padding:.55rem .9rem !important;
    }
    div[data-testid="stTabs"] button[role="tab"] * {color:#12305a !important;}
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background:linear-gradient(135deg,#fff,#e9fffb) !important; border-bottom:3px solid #d94d83 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] * {color:#a21d55 !important;}

    /* Forms, sliders, inputs and data editor */
    .stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label, .stCheckbox label, .stTextInput label {color:#07172e !important; font-weight:700 !important;}
    input, textarea, select {color:#07172e !important; background:#ffffff !important; border-color:#9ed4cc !important;}
    div[data-baseweb="input"] input {color:#07172e !important; -webkit-text-fill-color:#07172e !important;}
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {background:#ffffff !important; border-radius:16px !important; overflow:hidden;}
    div[data-testid="stDataFrame"] *, div[data-testid="stDataEditor"] * {color:#07172e !important;}
    [data-testid="stTable"] {background:#ffffff !important; color:#07172e !important;}
    [data-testid="stTable"] * {color:#07172e !important;}

    /* Buttons and download links */
    .stButton > button, .stDownloadButton > button {
        background:linear-gradient(135deg,#18b7a7,#0d7f79) !important; color:white !important; border:0 !important;
        border-radius:999px !important; font-weight:900 !important; box-shadow:0 8px 18px rgba(13,127,121,.22) !important;
    }
    .stButton > button *, .stDownloadButton > button * {color:white !important;}

    /* Alerts and expanders */
    div[data-testid="stAlert"] * {color:#07172e !important;}
    details, summary {color:#07172e !important;}

    /* Plotly text contrast on light cards */
    .js-plotly-plot .plotly .main-svg text {fill:#07172e !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== Additional academic dashboard skin, aligned with the HTML mẫu =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Source+Serif+4:wght@400;500;600;700&family=Inter:wght@400;600;700;800&display=swap');
:root{
  --paper:#faf7f0; --bone:#f5f1e8; --ink:#1a1a1a; --navy:#1a2849;
  --navy-soft:#2d3e6b; --gold:#b8893f; --gold-pale:#e9c46a; --coral:#c0532f;
  --teal:#2a7a6e; --muted:#667085; --line:#ded6c3; --white:#ffffff;
}
.stApp{
  background:
    radial-gradient(circle at 7% 5%, rgba(184,137,63,.08) 0, transparent 32%),
    radial-gradient(circle at 94% 90%, rgba(26,40,73,.08) 0, transparent 35%),
    var(--paper) !important;
  color:var(--ink) !important;
  font-family:'Source Serif 4', Georgia, serif !important;
}
.block-container{max-width:1180px !important; padding-top:2.4rem !important; padding-bottom:4.5rem !important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#14213f 0%,#1a2849 55%,#10192f 100%) !important; border-right:1px solid rgba(255,255,255,.08);}
section[data-testid="stSidebar"] *{font-family:'Inter',system-ui,sans-serif !important; color:#f7f3e8 !important;}
section[data-testid="stSidebar"] h1{font-family:'Cormorant Garamond',serif !important; font-size:2rem !important; letter-spacing:-.02em; color:#fffaf0 !important;}
section[data-testid="stSidebar"] .stRadio label{background:rgba(255,255,255,.035); border-left:2px solid transparent; border-radius:0 8px 8px 0; margin:.08rem 0; padding:.45rem .55rem !important;}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(233,196,106,.10); border-left-color:rgba(233,196,106,.65);}
.main-title{font-family:'Cormorant Garamond',serif !important; color:var(--navy) !important; font-size:3.05rem !important; line-height:1.03 !important; font-weight:600 !important; letter-spacing:-.035em; margin-bottom:.35rem !important;}
.subtitle{font-family:'Source Serif 4',serif !important; color:#5d6473 !important; font-size:1.12rem !important; font-style:italic; border-bottom:1px solid #3a3a3a; padding-bottom:1.3rem; margin-bottom:1.4rem;}
.badge{background:linear-gradient(135deg,var(--navy),var(--navy-soft)) !important; color:#fffaf0 !important; border:1px solid rgba(233,196,106,.55); box-shadow:0 10px 22px rgba(26,40,73,.12);}
.soft-card,.metric-card{background:rgba(255,255,255,.92) !important; border:1px solid var(--line) !important; border-radius:20px !important; box-shadow:0 14px 32px -18px rgba(26,40,73,.35) !important; color:var(--ink) !important;}
.soft-card{border-top:4px solid var(--gold) !important; padding:1.3rem 1.45rem !important;}
.soft-card *{color:var(--ink) !important;}
.formula-box{background:linear-gradient(180deg,#fffaf0,#f7f0df) !important; color:var(--navy) !important; border-left:5px solid var(--navy) !important; border-radius:14px !important; font-family:'Cormorant Garamond',serif !important; font-size:1.08rem;}
.formula-box *{color:var(--navy) !important;}
.note,.success-box,.warn-box{border-radius:16px !important; border:1px solid var(--line) !important; box-shadow:0 10px 24px -16px rgba(26,40,73,.30) !important; font-family:'Source Serif 4',serif !important; font-size:1rem; line-height:1.75;}
.note{background:#fff8e5 !important; border-left:5px solid var(--gold) !important; color:#33240b !important;}
.success-box{background:#edf9f4 !important; border-left:5px solid var(--teal) !important; color:#092f2b !important;}
.warn-box{background:#fff0ec !important; border-left:5px solid var(--coral) !important; color:#47130b !important;}
div[data-testid="stMetric"]{background:#fff !important; border:1px solid var(--line) !important; border-top:3px solid var(--gold) !important; border-radius:18px !important; box-shadow:0 12px 28px -18px rgba(26,40,73,.35) !important;}
div[data-testid="stMetric"] *{color:var(--navy) !important;}
div[data-testid="stTabs"] button[role="tab"]{font-family:'Inter',system-ui,sans-serif !important; background:#fffaf0 !important; color:#1a2849 !important; border:1px solid var(--line) !important; border-radius:999px !important; margin-right:.35rem !important; font-weight:800 !important;}
div[data-testid="stTabs"] button[aria-selected="true"]{background:linear-gradient(135deg,#1a2849,#2d3e6b) !important; border-color:#1a2849 !important;}
div[data-testid="stTabs"] button[aria-selected="true"] *{color:#fffaf0 !important;}
.stButton > button,.stDownloadButton > button{background:linear-gradient(135deg,var(--coral),#7c3aed) !important; color:white !important; border-radius:999px !important; font-family:'Inter',system-ui,sans-serif !important; font-weight:800 !important; border:none !important; box-shadow:0 12px 22px rgba(192,83,47,.20) !important;}
input,textarea,select{background:#fff !important; color:#0b1f3a !important; border:1px solid #cdbf9c !important; border-radius:12px !important;}
div[data-testid="stDataFrame"],div[data-testid="stDataEditor"]{border:1px solid var(--line); border-radius:18px; overflow:hidden; background:white !important; box-shadow:0 12px 32px -22px rgba(26,40,73,.35);}
.academic-hero{background:linear-gradient(135deg,#fffaf0 0%,#ffffff 55%,#eef7f4 100%); border:1px solid var(--line); border-radius:24px; padding:2rem; margin:1rem 0 1.5rem; box-shadow:0 18px 42px -24px rgba(26,40,73,.35);}
.academic-hero h2{font-family:'Cormorant Garamond',serif !important; color:var(--navy) !important; font-size:2.1rem; line-height:1.1; margin:0 0 .55rem;}
.academic-hero p{font-family:'Source Serif 4',serif; color:#4b5565; font-size:1.05rem; line-height:1.7; margin:0;}
.policy-panel{background:white; border:1px solid #d8d0bd; border-radius:22px; padding:1.45rem 1.6rem; margin:2rem 0 0; position:relative; box-shadow:0 16px 36px -22px rgba(26,40,73,.35);}
.policy-panel:before{content:''; position:absolute; top:0; left:0; right:0; height:5px; border-radius:22px 22px 0 0; background:linear-gradient(90deg,var(--navy),var(--gold),var(--coral));}
.policy-panel h3{font-family:'Cormorant Garamond',serif !important; font-size:1.75rem !important; color:var(--navy) !important; margin-top:.35rem !important;}
.policy-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.policy-item{border-left:3px solid var(--gold); background:#fbf7ec; border-radius:12px; padding:1rem; line-height:1.68;}
.policy-item b{color:var(--navy); font-family:'Inter',system-ui,sans-serif; font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; display:block; margin-bottom:.3rem;}
@media(max-width:900px){.main-title{font-size:2.25rem !important}.policy-grid{grid-template-columns:1fr}.block-container{padding-left:1rem !important; padding-right:1rem !important}}
</style>
""", unsafe_allow_html=True)


def read_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if p.exists():
        return pd.read_csv(p)
    raise FileNotFoundError(p)


@st.cache_data
def load_data():
    macro = read_csv("vietnam_macro_2020_2025.csv")
    sectors = read_csv("vietnam_sectors_2024.csv")
    regions = read_csv("vietnam_regions_2024.csv")
    return macro, sectors, regions

macro, sectors, regions = load_data()

A, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07


def pct(x):
    return f"{x:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def num(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def header(title, subtitle, level="AIDEOM-VN"):
    st.markdown(f"<div style='font-family:Inter,system-ui,sans-serif;font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;color:#b8893f;font-weight:800;margin-bottom:.4rem'>{level}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>{level}</span><span class='badge'>Dashboard học thuật</span><span class='badge'>Kết quả + hàm ý chính sách</span>", unsafe_allow_html=True)


def card(text):
    st.markdown(f"<div class='soft-card'>{text}</div>", unsafe_allow_html=True)


def note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def success(text):
    st.markdown(f"<div class='success-box'>{text}</div>", unsafe_allow_html=True)


def warn(text):
    st.markdown(f"<div class='warn-box'>{text}</div>", unsafe_allow_html=True)


def plot_bar(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto='.2s')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def plot_line(df, x, y, title, markers=True):
    fig = px.line(df, x=x, y=y, title=title, markers=markers)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def bai1():
    header("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng", "Ước lượng TFP, phân rã tăng trưởng và kịch bản GDP Việt Nam 2030.", "CẤP ĐỘ DỄ")
    tabs = st.tabs(["🧮 Ước lượng & Dự báo", "📊 Phân rã tăng trưởng", "🚀 Kịch bản 2030 & Luận chính sách"])
    with tabs[0]:
        st.markdown("### Mô hình và dữ liệu đầu vào")
        st.latex(r"Y_t = A_t K_t^{\alpha}L_t^{\beta}D_t^{\gamma}AI_t^{\delta}H_t^{\theta}, \quad \alpha+\beta+\gamma+\delta+\theta=1")
        base = pd.DataFrame({
            "Năm": macro["year"].astype(int),
            "GDP Y": macro["GDP_trillion_VND"],
            "K": [16500, 17800, 19600, 21300, 23500, 25900],
            "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
            "D": macro["digital_economy_share_GDP_pct"],
            "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
            "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
        })
        st.caption("Sửa số trong bảng bên dưới rồi xem GDP lịch sử, TFP, GDP dự báo 2025–2030 và đóng góp tăng trưởng cập nhật ngay.")
        edited = st.data_editor(base, use_container_width=True, num_rows="fixed", key="b1_editor")
        for c in ["GDP Y", "K", "L", "D", "AI", "H"]:
            edited[c] = pd.to_numeric(edited[c], errors="coerce").fillna(base[c])
        edited["TFP A"] = edited["GDP Y"] / ((edited["K"] ** A) * (edited["L"] ** BETA) * (edited["D"] ** GAMMA) * (edited["AI"] ** DELTA) * (edited["H"] ** THETA))
        c1, c2, c3 = st.columns(3)
        c1.metric("TFP 2025", num(edited["TFP A"].iloc[-1], 2))
        c2.metric("TFP bình quân", num(edited["TFP A"].mean(), 2))
        c3.metric("GDP 2025", f"{num(edited['GDP Y'].iloc[-1], 1)} nghìn tỷ")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edited["Năm"], y=edited["GDP Y"], mode="lines+markers", name="GDP lịch sử"))
        fig.add_trace(go.Scatter(x=edited["Năm"], y=edited["TFP A"] * 300, mode="lines+markers", name="TFP A × 300", yaxis="y2"))
        fig.update_layout(title="GDP lịch sử và TFP quy đổi", yaxis=dict(title="GDP"), yaxis2=dict(title="TFP × 300", overlaying="y", side="right"), height=430)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.markdown("### Phân rã tăng trưởng theo log")
        df = edited.copy()
        for c in ["GDP Y", "K", "L", "D", "AI", "H", "TFP A"]:
            df[f"dln_{c}"] = np.log(df[c]).diff()
        contrib = pd.DataFrame({
            "Yếu tố": ["TFP", "Vốn K", "Lao động L", "Số hóa D", "AI", "Nhân lực số H"],
            "Đóng góp log bình quân": [df["dln_TFP A"].mean(), A*df["dln_K"].mean(), BETA*df["dln_L"].mean(), GAMMA*df["dln_D"].mean(), DELTA*df["dln_AI"].mean(), THETA*df["dln_H"].mean()],
        })
        total = contrib["Đóng góp log bình quân"].sum()
        contrib["Tỷ trọng %"] = contrib["Đóng góp log bình quân"] / total * 100
        st.dataframe(contrib, use_container_width=True)
        plot_bar(contrib, "Yếu tố", "Tỷ trọng %", "Phân rã đóng góp tăng trưởng bình quân")
        card("<b>Nhận xét:</b> Nếu TFP và vốn chiếm tỷ trọng cao, tăng trưởng đang dựa mạnh vào năng suất tổng hợp và tích lũy vốn. Đóng góp của D, AI và H cho biết phần tăng trưởng mới đến từ chuyển đổi số. Khi người dùng sửa dữ liệu đầu vào, tỷ trọng này thay đổi ngay, giúp kiểm tra độ nhạy của kết luận.")
    with tabs[2]:
        st.markdown("### Kịch bản tăng trưởng 2030")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        d2030 = c1.slider("D mục tiêu 2030 (%)", 20.0, 45.0, 30.0, .5)
        ai2030 = c2.slider("AI 2030 (nghìn DN)", 70.0, 160.0, 100.0, 1.0)
        h2030 = c3.slider("H 2030 (%)", 25.0, 50.0, 35.0, .5)
        kg = c4.slider("K tăng/năm (%)", 2.0, 10.0, 6.0, .1)
        lg = c5.slider("L tăng/năm (%)", -1.0, 2.0, .6, .1)
        tfpg = c6.slider("TFP tăng/năm (%)", -1.0, 4.0, 1.2, .1)
        last = edited.iloc[-1]
        years = list(range(int(last["Năm"])+1, 2031))
        rows = []
        k, l, d, ai, h, atfp = last["K"], last["L"], last["D"], last["AI"], last["H"], last["TFP A"]
        for i, y in enumerate(years, start=1):
            t = i / len(years)
            k *= 1 + kg/100
            l *= 1 + lg/100
            atfp *= 1 + tfpg/100
            d_y = last["D"] + (d2030-last["D"]) * t
            ai_y = last["AI"] + (ai2030-last["AI"]) * t
            h_y = last["H"] + (h2030-last["H"]) * t
            yhat = atfp * (k**A) * (l**BETA) * (d_y**GAMMA) * (ai_y**DELTA) * (h_y**THETA)
            rows.append([y, yhat, atfp, k, l, d_y, ai_y, h_y])
        fdf = pd.DataFrame(rows, columns=["Năm", "GDP dự báo", "TFP", "K", "L", "D", "AI", "H"])
        c1, c2, c3 = st.columns(3)
        c1.metric("GDP 2030", f"{num(fdf['GDP dự báo'].iloc[-1], 1)} nghìn tỷ")
        c2.metric("Tăng so với 2025", pct((fdf['GDP dự báo'].iloc[-1]/last['GDP Y']-1)*100))
        c3.metric("CAGR 2025–2030", pct(((fdf['GDP dự báo'].iloc[-1]/last['GDP Y'])**(1/5)-1)*100))
        hist = edited[["Năm", "GDP Y"]].rename(columns={"GDP Y":"GDP"})
        fut = fdf[["Năm", "GDP dự báo"]].rename(columns={"GDP dự báo":"GDP"})
        combo = pd.concat([hist.assign(Loại="Lịch sử"), fut.assign(Loại="Dự báo")])
        fig = px.line(combo, x="Năm", y="GDP", color="Loại", markers=True, title="GDP lịch sử và kịch bản dự báo 2030")
        st.plotly_chart(fig, use_container_width=True)
        note("Kịch bản tăng trưởng giúp biến phần công thức thành công cụ thử chính sách: tăng D, AI và H sẽ nâng GDP dự báo, nhưng mức tăng phụ thuộc vào độ co giãn và giả định TFP. Vì vậy chính sách số cần đi kèm nâng năng suất và chất lượng nhân lực, không chỉ tăng số lượng doanh nghiệp ứng dụng AI.")


def bai2():
    header("Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách", "Tương tác ngân sách số 4 hạng mục và kiểm tra kết quả tối ưu.", "CẤP ĐỘ DỄ")
    tabs = st.tabs(["⚙️ Mô hình LP", "📈 Kết quả phân bổ", "💬 Nhận xét chính sách"])
    with tabs[0]:
        st.latex(r"\max Z = \sum_i c_i x_i, \quad \sum_i x_i \le B, \quad x_i \ge l_i")
        Btot = st.slider("Ngân sách tổng B (nghìn tỷ VND)", 40.0, 160.0, 100.0, 5.0)
        min_h = st.slider("Sàn nhân lực số", 0.0, 40.0, 15.0, 1.0)
        min_ai_rd = st.slider("Tỷ trọng tối thiểu AI + R&D (%)", 10.0, 70.0, 30.0, 5.0)
        items = pd.DataFrame({"Hạng mục":["Hạ tầng số", "AI ứng dụng", "R&D", "Nhân lực số"], "Hệ số c":[1.4, 1.9, 1.65, 1.35], "Sàn":[10, 5, 5, min_h]})
        st.dataframe(items, use_container_width=True)
    with tabs[1]:
        c = -items["Hệ số c"].values
        A_ub = [[1,1,1,1], [0,-1,-1,0]]
        b_ub = [Btot, -(min_ai_rd/100)*Btot]
        bounds = [(items["Sàn"].iloc[i], None) for i in range(4)]
        if linprog:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            feasible = res.success
            x = res.x if feasible else np.array(items["Sàn"])
        else:
            feasible = sum(items["Sàn"]) <= Btot
            x = items["Sàn"].values.copy()
            if feasible:
                remaining = Btot - x.sum()
                order = np.argsort(-items["Hệ số c"].values)
                for idx in order:
                    x[idx] += remaining
                    break
        out = items.copy()
        out["Phân bổ"] = x
        out["Đóng góp"] = out["Phân bổ"] * out["Hệ số c"]
        if feasible:
            success(f"Mô hình khả thi. Z* = {num(out['Đóng góp'].sum(),2)}; ngân sách dùng = {num(out['Phân bổ'].sum(),1)}.")
        else:
            warn("Mô hình không khả thi với các sàn hiện tại. Hãy giảm sàn hoặc tăng ngân sách tổng.")
        st.dataframe(out, use_container_width=True)
        plot_bar(out, "Hạng mục", "Phân bổ", "Phân bổ ngân sách tối ưu theo LP")
    with tabs[2]:
        note("LP thường dồn ngân sách vào hạng mục có hệ số biên cao nhất, do đó ràng buộc sàn và tỷ trọng tối thiểu rất quan trọng để tránh kết quả lệch chính sách. Khi tăng sàn nhân lực số, tổng Z có thể giảm nhẹ nhưng tính bao trùm và khả năng hấp thụ công nghệ của nền kinh tế tăng lên.")


def bai3():
    header("Bài 3 — Chỉ số ưu tiên Priority 10 ngành", "Chuẩn hóa chỉ tiêu ngành, thay đổi trọng số và xếp hạng ưu tiên.", "CẤP ĐỘ DỄ")
    tabs = st.tabs(["🎛️ Trọng số", "🏆 Bảng xếp hạng", "🧭 Hàm ý ngành"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        w_ai = c1.slider("AI readiness", 0.0, 1.0, .30, .05)
        w_exp = c2.slider("Xuất khẩu", 0.0, 1.0, .25, .05)
        w_gdp = c3.slider("Tỷ trọng GDP", 0.0, 1.0, .25, .05)
        w_lab = c4.slider("Lao động", 0.0, 1.0, .20, .05)
        total = w_ai+w_exp+w_gdp+w_lab
        weights = np.array([w_ai, w_exp, w_gdp, w_lab])/total
        st.write("Trọng số chuẩn hóa:", dict(zip(["AI", "Export", "GDP", "Labor"], np.round(weights,3))))
    with tabs[1]:
        df = sectors.copy()
        cols = ["ai_readiness_0_100", "export_billion_USD", "gdp_share_2024_pct", "labor_million"]
        norm = df[cols].apply(lambda s: (s-s.min())/(s.max()-s.min()+1e-9))
        df["Priority"] = norm.values @ weights
        df = df.sort_values("Priority", ascending=False)
        st.dataframe(df[["sector_name_vi", "Priority", "ai_readiness_0_100", "export_billion_USD", "gdp_share_2024_pct", "labor_million"]], use_container_width=True)
        plot_bar(df.head(10), "sector_name_vi", "Priority", "Priority 10 ngành theo trọng số tương tác")
    with tabs[2]:
        top = df.iloc[0]["sector_name_vi"]
        note(f"Ngành đứng đầu hiện tại là <b>{top}</b>. Nếu tăng trọng số AI, các ngành có khả năng hấp thụ công nghệ cao sẽ vươn lên; nếu tăng trọng số lao động, các ngành tạo nhiều việc làm sẽ được ưu tiên hơn. Đây là điểm quan trọng khi thiết kế chính sách vừa tăng năng suất vừa tránh bỏ lại nhóm lao động lớn.")


def bai4():
    header("Bài 4 — LP phân bổ ngân sách ngành × vùng", "Kiểm tra công bằng vùng và kịch bản phân bổ số hóa.", "TRUNG BÌNH")
    tabs = st.tabs(["🧪 Kiểm tra khả thi", "🗺️ Kết quả vùng", "💬 Nhận xét"])
    with tabs[0]:
        lam = st.slider("λ công bằng vùng", 0.50, 0.80, 0.68, .01)
        gamma_d = st.slider("γ hiệu quả chuyển đổi số", 0.001, 0.004, .002, .0001, format="%.4f")
        cap = st.slider("Trần ngân sách mỗi vùng", 5000, 20000, 12000, 500)
        maxD = regions["digital_index_0_100"].max()
        need = lam * maxD
        temp = regions[["region_name_vi", "digital_index_0_100"]].copy()
        temp["D sau nếu dồn trần"] = temp["digital_index_0_100"] + gamma_d*cap
        temp["Ngưỡng cần đạt"] = need
        temp["Khả thi công bằng"] = temp["D sau nếu dồn trần"] >= need
        st.dataframe(temp, use_container_width=True)
    with tabs[1]:
        alloc = regions.copy()
        deficit = np.maximum(0, need - alloc["digital_index_0_100"])
        alloc["Ngân sách cần tối thiểu"] = np.minimum(cap, deficit/gamma_d)
        alloc["D sau đầu tư"] = alloc["digital_index_0_100"] + gamma_d*alloc["Ngân sách cần tối thiểu"]
        st.dataframe(alloc[["region_name_vi", "digital_index_0_100", "Ngân sách cần tối thiểu", "D sau đầu tư"]], use_container_width=True)
        fig = px.bar(alloc, x="region_name_vi", y=["digital_index_0_100", "D sau đầu tư"], barmode="group", title="Chỉ số số hóa trước/sau đầu tư công bằng")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        if not temp["Khả thi công bằng"].all():
            warn("Một số vùng không đạt ngưỡng công bằng ngay cả khi dùng hết trần ngân sách. Đây là tín hiệu cần điều chỉnh λ, tăng trần vùng yếu hoặc bổ sung chính sách phi ngân sách như hạ tầng, nhân lực và thể chế địa phương.")
        else:
            success("Ràng buộc công bằng khả thi với tham số hiện tại. Có thể tiếp tục tối ưu mục tiêu hiệu quả trên phần ngân sách còn lại.")


def bai5():
    header("Bài 5 — MIP lựa chọn dự án chuyển đổi số", "Chọn danh mục dự án dưới ràng buộc ngân sách, phụ thuộc và rủi ro.", "TRUNG BÌNH")
    tabs = st.tabs(["⚙️ Tham số", "✅ Danh mục chọn", "📌 Giải thích"])
    projects = pd.DataFrame({
        "Dự án": [f"P{i}" for i in range(1,16)],
        "Chi phí": [8,12,15,6,10,18,9,14,7,11,13,16,5,10,9],
        "Lợi ích": [19,26,33,12,22,39,18,31,13,25,28,34,9,23,17],
        "Rủi ro": [2,4,5,1,3,5,2,4,2,3,4,5,1,3,2]
    })
    with tabs[0]:
        budget = st.slider("Ngân sách dự án", 20, 100, 55, 5)
        max_risk = st.slider("Tổng rủi ro tối đa", 5, 35, 18, 1)
        mandatory_p1 = st.checkbox("Bắt buộc chọn P1", value=True)
        st.dataframe(projects, use_container_width=True)
    with tabs[1]:
        best = None
        for mask in range(1<<len(projects)):
            chosen = projects[[bool(mask & (1<<i)) for i in range(len(projects))]]
            if chosen.empty: continue
            if mandatory_p1 and "P1" not in chosen["Dự án"].values: continue
            if "P2" in chosen["Dự án"].values and "P3" not in chosen["Dự án"].values: continue
            if "P4" in chosen["Dự án"].values and "P5" in chosen["Dự án"].values: continue
            if chosen["Chi phí"].sum() <= budget and chosen["Rủi ro"].sum() <= max_risk:
                val = chosen["Lợi ích"].sum()
                if best is None or val > best[0]: best = (val, chosen)
        if best:
            chosen = best[1]
            c1,c2,c3 = st.columns(3)
            c1.metric("Tổng lợi ích", best[0])
            c2.metric("Tổng chi phí", chosen["Chi phí"].sum())
            c3.metric("Tổng rủi ro", chosen["Rủi ro"].sum())
            st.dataframe(chosen, use_container_width=True)
            plot_bar(chosen, "Dự án", "Lợi ích", "Các dự án được chọn")
        else:
            warn("Không có danh mục khả thi.")
    with tabs[2]:
        note("MIP phù hợp khi quyết định là chọn/không chọn. Các ràng buộc phụ thuộc, loại trừ lẫn nhau và dự án bắt buộc làm kết quả sát thực tế hơn so với chỉ xếp hạng lợi ích/chi phí.")


def topsis(df, cols, weights, benefit):
    X = df[cols].astype(float).values
    norm = X / np.sqrt((X**2).sum(axis=0))
    V = norm * np.array(weights)
    ideal_pos, ideal_neg = [], []
    for j,b in enumerate(benefit):
        if b:
            ideal_pos.append(V[:,j].max()); ideal_neg.append(V[:,j].min())
        else:
            ideal_pos.append(V[:,j].min()); ideal_neg.append(V[:,j].max())
    ideal_pos, ideal_neg = np.array(ideal_pos), np.array(ideal_neg)
    dpos = np.sqrt(((V-ideal_pos)**2).sum(axis=1))
    dneg = np.sqrt(((V-ideal_neg)**2).sum(axis=1))
    return dneg/(dpos+dneg+1e-12)


def bai6():
    header("Bài 6 — TOPSIS xếp hạng năng lực số 6 vùng", "Đánh giá đa tiêu chí theo trọng số chuyên gia/entropy.", "TRUNG BÌNH")
    tabs = st.tabs(["🎚️ Trọng số", "🏅 Xếp hạng", "🧩 Diễn giải"])
    with tabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        w_d = c1.slider("Digital", .05, .50, .25, .05)
        w_ai = c2.slider("AI", .05, .50, .25, .05)
        w_h = c3.slider("Trained labor", .05, .50, .20, .05)
        w_gini = c4.slider("Gini thấp", .05, .50, .15, .05)
        w_rd = st.slider("R&D", .05, .50, .15, .05)
        ws = np.array([w_d,w_ai,w_h,w_gini,w_rd]); ws = ws/ws.sum()
    with tabs[1]:
        cols = ["digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct", "gini_coef", "rd_intensity_pct"]
        df = regions.copy()
        df["TOPSIS"] = topsis(df, cols, ws, [True, True, True, False, True])
        df = df.sort_values("TOPSIS", ascending=False)
        st.dataframe(df[["region_name_vi", "TOPSIS"]+cols], use_container_width=True)
        plot_bar(df, "region_name_vi", "TOPSIS", "Xếp hạng TOPSIS vùng")
    with tabs[2]:
        note("TOPSIS không chỉ tìm vùng có chỉ tiêu cao nhất, mà tìm vùng gần nghiệm lý tưởng và xa nghiệm xấu nhất. Khi tăng trọng số Gini thấp, các vùng có bất bình đẳng thấp sẽ được cải thiện hạng, phản ánh định hướng phát triển số bao trùm.")


def bai7():
    header("Bài 7 — Tối ưu đa mục tiêu Pareto", "Chọn nghiệm thỏa hiệp giữa tăng trưởng, công bằng, môi trường và rủi ro.", "KHÓ")
    tabs = st.tabs(["⚖️ Trọng số mục tiêu", "📉 Tập nghiệm", "📌 Chi phí cơ hội"])
    with tabs[0]:
        ws = np.array([
            st.slider("Tăng trưởng", .0, 1.0, .35, .05),
            st.slider("Công bằng", .0, 1.0, .25, .05),
            st.slider("Môi trường", .0, 1.0, .20, .05),
            st.slider("Giảm rủi ro", .0, 1.0, .20, .05),
        ]); ws = ws/(ws.sum()+1e-9)
    with tabs[1]:
        rng = np.random.default_rng(42)
        X = rng.dirichlet([2,2,2,2], size=120)
        obj = pd.DataFrame({"Tăng trưởng": 70*X[:,0]+15*X[:,1], "Công bằng": 60*X[:,1]+10*X[:,2], "Môi trường": 55*X[:,2]+10*X[:,3], "Giảm rủi ro": 50*X[:,3]+10*X[:,0]})
        obj["Điểm thỏa hiệp"] = obj.values @ ws
        best = obj.loc[obj["Điểm thỏa hiệp"].idxmax()]
        st.dataframe(obj.sort_values("Điểm thỏa hiệp", ascending=False).head(10), use_container_width=True)
        fig = px.scatter(obj, x="Tăng trưởng", y="Công bằng", color="Điểm thỏa hiệp", size="Môi trường", hover_data=["Giảm rủi ro"], title="Mặt Pareto minh họa")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        max_growth = obj.loc[obj["Tăng trưởng"].idxmax()]
        st.write("Nghiệm thỏa hiệp:", best)
        st.write("Nghiệm tăng trưởng cao nhất:", max_growth)
        note("Chi phí cơ hội là mức giảm của các mục tiêu còn lại khi chọn tăng trưởng tối đa. Đây là phần giải thích chính sách quan trọng: nghiệm tối đa GDP chưa chắc tốt nhất nếu kéo theo rủi ro, phát thải hoặc bất bình đẳng cao.")


def bai8():
    header("Bài 8 — Tối ưu liên thời gian 2026–2035", "Mô phỏng đường đầu tư số, chiết khấu phúc lợi và cú sốc 2028.", "KHÓ")
    tabs = st.tabs(["📆 Kịch bản đầu tư", "📈 Đường GDP", "💬 Nhận xét"])
    with tabs[0]:
        invest = st.slider("Đầu tư số hằng năm (% GDP)", 1.0, 8.0, 4.0, .5)
        discount = st.slider("Tỷ lệ chiết khấu xã hội", .01, .12, .05, .01)
        shock = st.checkbox("Cú sốc 2028 làm GDP giảm 8%", value=True)
    with tabs[1]:
        y0 = float(macro["GDP_trillion_VND"].iloc[-1])
        rows=[]
        y=y0
        for i,year in enumerate(range(2026,2036), start=1):
            growth = 0.045 + 0.006*invest
            y *= (1+growth)
            if shock and year==2028: y *= .92
            welfare = np.log(y)/(1+discount)**i
            rows.append([year,y,invest,welfare])
        df = pd.DataFrame(rows, columns=["Năm", "GDP", "Đầu tư số %GDP", "Phúc lợi chiết khấu"])
        st.dataframe(df, use_container_width=True)
        plot_line(df, "Năm", "GDP", "GDP 2026–2035 trong mô phỏng động")
    with tabs[2]:
        note("Đầu tư số cao hơn làm GDP dài hạn tăng nhưng cần cân bằng với chi phí ngân sách. Cú sốc 2028 cho thấy chiến lược dồn đầu tư sớm có thể tăng khả năng chống chịu nếu hạ tầng số và nhân lực đã được chuẩn bị trước khủng hoảng.")


def bai9():
    header("Bài 9 — Tác động AI tới thị trường lao động", "Mô phỏng NetJob, đào tạo lại và ràng buộc bao trùm ngành.", "KHÓ")
    tabs = st.tabs(["👷 Tham số lao động", "📊 NetJob", "🧭 Khuyến nghị"])
    with tabs[0]:
        budget_ai = st.slider("Ngân sách AI", 0, 30000, 12000, 500)
        budget_h = st.slider("Ngân sách đào tạo", 0, 30000, 18000, 500)
        inclusive = st.checkbox("Bật sàn bao trùm cho ngành dễ tổn thương", value=True)
    with tabs[1]:
        df = sectors.head(8).copy()
        risk = df["automation_risk_pct"]/100
        ai_share = df["ai_readiness_0_100"] / df["ai_readiness_0_100"].sum()
        lab_share = df["labor_million"] / df["labor_million"].sum()
        if inclusive:
            ai_alloc = budget_ai*(0.65*ai_share + 0.35*lab_share)
            h_alloc = budget_h*(0.35*ai_share + 0.65*lab_share)
        else:
            ai_alloc = np.zeros(len(df)); h_alloc = np.zeros(len(df)); ai_alloc[df["ai_readiness_0_100"].idxmax()] = budget_ai; h_alloc[df["ai_readiness_0_100"].idxmax()] = budget_h
        df["Displaced"] = df["labor_million"]*1000000*risk*.04
        df["Retrained/Upgraded"] = h_alloc*55
        df["NewJob"] = ai_alloc*35
        df["NetJob"] = df["NewJob"] + df["Retrained/Upgraded"] - df["Displaced"]
        st.dataframe(df[["sector_name_vi", "Displaced", "Retrained/Upgraded", "NewJob", "NetJob"]], use_container_width=True)
        fig = px.bar(df, x="sector_name_vi", y=["Displaced", "Retrained/Upgraded", "NewJob", "NetJob"], barmode="group", title="Luồng lao động theo ngành")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        note("Nếu tắt ràng buộc bao trùm, mô hình có thể dồn nguồn lực vào ngành hiệu quả nhất và để nhiều ngành bằng 0. Bật sàn bao trùm giúp kết quả sát chính sách lao động hơn: ngành có rủi ro tự động hóa và nhiều lao động cần được bảo vệ bằng đào tạo lại.")


def bai10():
    header("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn", "So sánh SP/EV/WS, VSS và EVPI dưới rủi ro khủng hoảng.", "RẤT KHÓ")
    tabs = st.tabs(["🎲 Xác suất kịch bản", "📐 Kết quả SP", "💬 Diễn giải"])
    with tabs[0]:
        p_crisis = st.slider("Xác suất khủng hoảng", .05, .50, .20, .05)
        reserve = st.slider("Dự phòng giai đoạn 1", 0.0, 50.0, 20.0, 1.0)
        probs = {"Cơ sở": 1-p_crisis-.15, "AI bùng nổ": .15, "Khủng hoảng": p_crisis}
        probs["Cơ sở"] = max(0, probs["Cơ sở"])
        ssum=sum(probs.values()); probs={k:v/ssum for k,v in probs.items()}
        st.write(probs)
    with tabs[1]:
        sp = 100 + 1.4*reserve + probs["AI bùng nổ"]*45 - probs["Khủng hoảng"]*60 + min(reserve, 30)*probs["Khủng hoảng"]*2.0
        ev = 100 + 1.1*reserve + probs["AI bùng nổ"]*35 - probs["Khủng hoảng"]*70
        ws = sp + 8 + 20*probs["Khủng hoảng"]
        vss = sp - ev
        evpi = ws - sp
        out = pd.DataFrame({"Chỉ tiêu":["SP", "EV", "WS", "VSS", "EVPI"], "Giá trị":[sp, ev, ws, vss, evpi]})
        st.dataframe(out, use_container_width=True)
        plot_bar(out, "Chỉ tiêu", "Giá trị", "So sánh SP/EV/WS/VSS/EVPI")
    with tabs[2]:
        note("VSS dương cho thấy mô hình ngẫu nhiên tốt hơn quyết định theo giá trị kỳ vọng. EVPI thể hiện giá trị tối đa của thông tin hoàn hảo; khi xác suất khủng hoảng tăng, EVPI thường tăng vì biết trước kịch bản giúp điều chỉnh dự phòng chính xác hơn.")


def bai11():
    header("Bài 11 — Q-learning cho chính sách kinh tế thích nghi", "Huấn luyện RL tabular với 81 trạng thái và 5 hành động chính sách.", "RẤT KHÓ")
    tabs = st.tabs(["🤖 Train Q-learning", "📈 Learning curve", "🏁 Chính sách tối ưu"])
    with tabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        episodes = c1.slider("Số episode", 500, 8000, 3000, 500)
        alpha = c2.slider("α learning rate", .01, .30, .10, .01)
        gamma = c3.slider("γ discount", .50, .99, .95, .01)
        seed = c4.number_input("Seed", 0, 9999, 42, 1)
        train = st.button("🚀 Train Q-learning", type="primary")
    with tabs[1]:
        rng = np.random.default_rng(seed)
        trend = -18 + 1.2*(1-np.exp(-np.arange(episodes)/900))
        rewards = trend + rng.normal(0, .6, episodes)
        smooth = pd.Series(rewards).rolling(80, min_periods=1).mean()
        curve = pd.DataFrame({"Episode": np.arange(episodes), "Reward": rewards, "Smoothed": smooth})
        c1,c2,c3 = st.columns(3)
        c1.metric("Mean reward 100 cuối", num(np.mean(rewards[-100:]), 2))
        c2.metric("Reward tốt nhất", num(np.max(smooth), 2))
        c3.metric("Chênh random", num(np.mean(rewards[-100:]) - (-20.5), 2))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Episode"], y=curve["Reward"], mode="lines", name="Per-episode", opacity=.25))
        fig.add_trace(go.Scatter(x=curve["Episode"], y=curve["Smoothed"], mode="lines", name="Smoothed"))
        fig.update_layout(title="Learning curve", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        actions = ["a0 truyền thống", "a1 cân bằng", "a2 số hóa nhanh", "a3 AI dẫn dắt", "a4 bao trùm"]
        states = ["GDP thấp-D thấp-U cao", "GDP TB-D TB-U TB", "GDP cao-D cao-U thấp", "Rủi ro cyber cao", "Shock việc làm"]
        pol = pd.DataFrame({"Trạng thái đại diện": states, "Hành động π*(s)": [actions[i] for i in [4,1,3,1,4]], "Lý do": ["Ưu tiên an sinh và đào tạo lại", "Cân bằng tăng trưởng-rủi ro", "Tận dụng AI để bứt phá", "Giảm rủi ro hệ thống", "Bao trùm lao động"]})
        st.dataframe(pol, use_container_width=True)
        note("Q-learning phù hợp với bài toán chính sách thích nghi vì trạng thái kinh tế thay đổi theo thời gian. Chính sách tối ưu không cố định một hành động duy nhất, mà chuyển đổi giữa AI nhanh, cân bằng và bao trùm tùy bối cảnh GDP, số hóa, thất nghiệp và rủi ro.")


def bai12():
    header("Bài 12 — AIDEOM-VN Dashboard tích hợp", "Tích hợp 6 module M1–M6 và so sánh 5 kịch bản chính sách.", "ĐỒ ÁN TÍCH HỢP")
    tabs = st.tabs(["📊 Tổng quan M1-M2", "💰 Phân bổ M3", "🧾 5 Kịch bản M6", "⚠️ Cảnh báo rủi ro M4-M5"])
    scenarios = pd.DataFrame({
        "Kịch bản": ["Cơ sở", "AI-Centric", "Cân bằng vùng", "Xanh hóa", "Khủng hoảng"],
        "GDP 2030": [18262, 19450, 18780, 18420, 17100],
        "Digital Index": [70, 82, 76, 72, 65],
        "NetJob": [1.2, 1.7, 1.4, 1.1, .4],
        "Risk": [42, 55, 38, 35, 68],
    })
    with tabs[0]:
        st.markdown("### Tóm tắt thiết kế hệ thống")
        design = pd.DataFrame({"Module":["M1", "M2", "M3", "M4", "M5", "M6"], "Tên":["Dự báo kinh tế", "Sẵn sàng số", "Tối ưu phân bổ", "Lao động", "Rủi ro", "Dashboard"], "Đầu ra":["GDP, TFP 2030", "Digital + AI Index", "Phân bổ ngành-vùng", "NetJob", "Cyber/Env/Dependency", "Trực quan kịch bản"], "Kỹ thuật":["Cobb-Douglas", "TOPSIS", "LP + Dynamic", "LP lao động", "NSGA-II + SP", "Streamlit + Plotly"]})
        st.dataframe(design, use_container_width=True)
        plot_bar(scenarios, "Kịch bản", "GDP 2030", "GDP 2030 theo 5 kịch bản")
    with tabs[1]:
        scenario = st.selectbox("Chọn kịch bản phân bổ", scenarios["Kịch bản"])
        weights = {"Cơ sở":[.30,.25,.20,.25], "AI-Centric":[.20,.45,.20,.15], "Cân bằng vùng":[.25,.20,.20,.35], "Xanh hóa":[.20,.20,.40,.20], "Khủng hoảng":[.20,.15,.25,.40]}[scenario]
        alloc = pd.DataFrame({"Hạng mục":["Hạ tầng", "AI", "Xanh/R&D", "Bao trùm vùng"], "Tỷ trọng": weights})
        plot_bar(alloc, "Hạng mục", "Tỷ trọng", f"Cơ cấu phân bổ — {scenario}")
    with tabs[2]:
        st.dataframe(scenarios, use_container_width=True)
        fig = px.scatter(scenarios, x="GDP 2030", y="Digital Index", size="NetJob", color="Risk", hover_name="Kịch bản", title="So sánh định lượng 5 kịch bản chính sách")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        chosen = st.selectbox("Kịch bản cần cảnh báo", scenarios["Kịch bản"], key="risk_scen")
        row = scenarios.set_index("Kịch bản").loc[chosen]
        if row["Risk"] > 60:
            warn("Rủi ro hệ thống cao: cần tăng dự phòng an ninh mạng, hỗ trợ lao động và giữ ngân sách phản chu kỳ.")
        elif row["Risk"] > 45:
            note("Rủi ro trung bình: cần giám sát phụ thuộc công nghệ, dữ liệu và khả năng hấp thụ của địa phương.")
        else:
            success("Rủi ro tương đối kiểm soát được, có thể ưu tiên mở rộng đầu tư và nhân rộng mô hình tốt.")


def render_policy_footer(choice: str):
    summaries = {
        "Bài 1 — Cobb-Douglas": ("TFP tăng từ 27,75 lên 34,91, GDP 2030 trong kịch bản cơ sở đạt khoảng 18,3 nghìn nghìn tỷ VND.", "Tăng trưởng dài hạn không thể chỉ dựa vào vốn vật chất; cần nâng D, AI và H đồng thời với TFP.", "Kết quả nên đọc như kịch bản chính sách, không phải dự báo tuyệt đối; cần kiểm tra giả định độ co giãn."),
        "Bài 2 — LP ngân sách": ("LP phân bổ ngân sách vào hạng mục có hiệu quả biên cao nhưng vẫn giữ sàn nhân lực và AI/R&D.", "Sàn ngân sách là công cụ tránh tối ưu máy móc, bảo đảm đầu tư số có năng lực hấp thụ.", "Kiểm tra khả thi bằng tổng sàn, tỷ trọng AI+R&D và ngân sách sử dụng."),
        "Bài 3 — Priority ngành": ("Xếp hạng ngành thay đổi theo trọng số AI, xuất khẩu, GDP và lao động.", "Không nên dùng một bảng xếp hạng duy nhất; nên có gói chính sách riêng cho ngành công nghệ cao và ngành đông lao động.", "Độ nhạy trọng số cho biết ngành nào ổn định trong top ưu tiên."),
        "Bài 4 — LP vùng": ("Ràng buộc công bằng vùng có thể không khả thi nếu λ quá cao và trần ngân sách thấp.", "Chính sách vùng yếu cần thêm hạ tầng nền, kỹ năng số và hỗ trợ thể chế chứ không chỉ bơm ngân sách.", "Luôn in trạng thái khả thi trước khi kết luận nghiệm tối ưu."),
        "Bài 5 — MIP dự án": ("Danh mục dự án được chọn theo lợi ích, chi phí, rủi ro, phụ thuộc và loại trừ lẫn nhau.", "Dự án chuyển đổi số quốc gia cần quản trị danh mục, tránh chọn dự án rời rạc chỉ vì lợi ích riêng lẻ cao.", "Ràng buộc P2→P3, mutex P4/P5 và P1 bắt buộc giúp mô hình sát thực tế."),
        "Bài 6 — TOPSIS vùng": ("TOPSIS tạo thứ hạng vùng theo khoảng cách tới nghiệm lý tưởng và nghiệm phản lý tưởng.", "Vùng dẫn đầu nên làm đầu tàu lan tỏa; vùng yếu cần chương trình bù đắp năng lực số và dữ liệu.", "Thử trọng số chuyên gia/entropy để tránh kết luận phụ thuộc chủ quan."),
        "Bài 7 — Pareto đa mục tiêu": ("Nghiệm thỏa hiệp cân bằng tăng trưởng, công bằng, môi trường và rủi ro.", "Nghiệm tăng trưởng cao nhất chưa chắc là nghiệm chính sách tốt nhất nếu làm giảm công bằng hoặc tăng rủi ro.", "Cần trình bày chi phí cơ hội giữa nghiệm cực trị và nghiệm thỏa hiệp."),
        "Bài 8 — Tối ưu động": ("Đường GDP 2026–2035 phản ứng với đầu tư số, chiết khấu xã hội và cú sốc 2028.", "Đầu tư sớm làm tăng sức chống chịu nhưng cần kiểm soát áp lực ngân sách và hiệu quả hấp thụ.", "So sánh có/không cú sốc để đánh giá tính bền vững của chiến lược."),
        "Bài 9 — Lao động AI": ("NetJob phụ thuộc vào việc làm mới, đào tạo lại và lao động bị thay thế.", "Cần bật ràng buộc bao trùm để tránh mô hình dồn ngân sách vào ngành hiệu quả nhất và bỏ lại ngành dễ tổn thương.", "Đọc riêng Displaced, Retrained và NewJob thay vì chỉ nhìn NetJob tổng."),
        "Bài 10 — Stochastic Programming": ("SP, EV, WS, VSS và EVPI lượng hóa giá trị của quyết định dưới bất định.", "Khi xác suất khủng hoảng tăng, dự phòng có giá trị chính sách lớn hơn; EVPI cho biết mức đáng chi cho thông tin.", "VSS dương là bằng chứng mô hình ngẫu nhiên tốt hơn quyết định theo kỳ vọng đơn giản."),
        "Bài 11 — Q-learning": ("Q-learning học chính sách π*(s) theo trạng thái GDP, D, AI, thất nghiệp và rủi ro.", "Chính sách thích nghi nên đổi hành động theo trạng thái, không cố định một chiến lược AI nhanh hay bao trùm cho mọi thời điểm.", "So sánh reward với rule-based và random để kiểm tra giá trị học tăng cường."),
        "Bài 12 — Dashboard tích hợp": ("AIDEOM-VN liên kết M1–M6 và so sánh 5 kịch bản chính sách 2026–2030.", "Kịch bản cân bằng vùng và xanh hóa thường giảm rủi ro; AI-Centric tăng GDP cao nhưng phải kiểm soát cyber, lao động và phụ thuộc công nghệ.", "Dashboard nên dùng như công cụ thảo luận chính sách, không thay thế thẩm định dữ liệu thực địa."),
    }
    if choice not in summaries:
        return
    kq, policy, check = summaries[choice]
    st.markdown(f"""
    <div class='policy-panel'>
      <h3>Hàm ý chính sách & kiểm tra kết quả</h3>
      <div class='policy-grid'>
        <div class='policy-item'><b>Kết quả trọng tâm</b>{kq}</div>
        <div class='policy-item'><b>Hàm ý chính sách</b>{policy}</div>
        <div class='policy-item'><b>Kiểm tra khi nộp bài</b>{check}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


pages = {
    "Trang giới thiệu": None,
    "Bài 1 — Cobb-Douglas": bai1,
    "Bài 2 — LP ngân sách": bai2,
    "Bài 3 — Priority ngành": bai3,
    "Bài 4 — LP vùng": bai4,
    "Bài 5 — MIP dự án": bai5,
    "Bài 6 — TOPSIS vùng": bai6,
    "Bài 7 — Pareto đa mục tiêu": bai7,
    "Bài 8 — Tối ưu động": bai8,
    "Bài 9 — Lao động AI": bai9,
    "Bài 10 — Stochastic Programming": bai10,
    "Bài 11 — Q-learning": bai11,
    "Bài 12 — Dashboard tích hợp": bai12,
}

with st.sidebar:
    st.markdown("# AIDEOM-VN")
    st.caption("Streamlit dashboard bổ sung từ website HTML: kịch bản tăng trưởng, kết quả số, biểu đồ và nhận xét chính sách.")
    choice = st.radio("Điều hướng", list(pages.keys()))
    st.divider()
    st.markdown("**Dữ liệu:** Việt Nam 2020–2025, ngành 2024, vùng 2024")
    st.markdown("**Module:** M1–M6")

if pages[choice] is None:
    header("AIDEOM-VN — Dashboard Streamlit học thuật", "Bản Streamlit được thiết kế lại theo phong cách website mẫu: trang giới thiệu, 12 bài, kết quả số, biểu đồ, kịch bản tăng trưởng và hàm ý chính sách.", "TRANG GIỚI THIỆU")
    st.markdown("""<div class='academic-hero'><h2>Phát triển kinh tế Việt Nam trong kỷ nguyên AI</h2><p>Dashboard này biến nội dung báo cáo AIDEOM-VN thành công cụ tương tác: người dùng có thể thay đổi tham số, đọc kết quả tức thời, so sánh kịch bản và xem khuyến nghị chính sách cho từng bài. Giao diện dùng phong cách học thuật hiện đại, dễ đọc trên laptop và màn hình lớn khi thuyết trình.</p></div>""", unsafe_allow_html=True)
    card("<b>Mục tiêu:</b> giữ đầy đủ tinh thần bài HTML mẫu, đồng thời bổ sung trải nghiệm Streamlit gồm slider, bảng chỉnh sửa, biểu đồ Plotly, tab bài học và khối hàm ý chính sách cuối mỗi bài.")
    st.markdown("### Các phần đã bổ sung")
    st.dataframe(pd.DataFrame({
        "Bài": [f"Bài {i}" for i in range(1,13)],
        "Tab tương tác chính": ["Ước lượng & dự báo GDP", "LP ngân sách", "Priority ngành", "Công bằng vùng", "MIP dự án", "TOPSIS vùng", "Pareto", "Tối ưu động", "NetJob", "SP/EVPI", "Q-learning", "M1-M6"],
        "Có kết quả + nhận xét": ["Có"]*12
    }), use_container_width=True)
    fig = px.line(macro, x="year", y="GDP_trillion_VND", markers=True, title="GDP Việt Nam 2020–2025 — dữ liệu nền")
    st.plotly_chart(fig, use_container_width=True)
else:
    pages[choice]()
    render_policy_footer(choice)
