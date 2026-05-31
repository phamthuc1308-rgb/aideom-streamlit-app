import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(page_title="AIDEOM-VN Premium", page_icon="🇻🇳", layout="wide")

# =========================================================
# PREMIUM EDITORIAL THEME — inspired by user's HTML reference
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --ink:#1a1a1a; --bone:#f5f1e8; --paper:#faf7f0; --navy:#1a2849;
  --navy-soft:#2d3e6b; --gold:#b8893f; --gold-pale:#e9c46a; --coral:#c0532f;
  --rule:#3a3a3a; --muted:#6b6b6b; --teal:#2a7a6e;
}
html, body, [class*="css"]{font-family:'Source Serif 4', Georgia, serif; color:var(--ink);}
.stApp{background: radial-gradient(circle at 10% 3%, rgba(184,137,63,.06) 0, transparent 30%), radial-gradient(circle at 95% 90%, rgba(26,40,73,.06) 0, transparent 35%), var(--paper);}
.block-container{max-width:1180px; padding-top:2.2rem; padding-bottom:4rem;}
[data-testid="stSidebar"]{background:var(--navy); border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] *{font-family:'Inter', system-ui, sans-serif; color:var(--bone)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Cormorant Garamond',serif!important; letter-spacing:-.01em;}
[data-testid="stSidebar"] .stRadio label{padding:.25rem .35rem; border-radius:8px;}
[data-testid="stSidebar"] .stRadio label:hover{background:rgba(233,196,106,.08)}
.hero{padding:38px 42px 34px; border:1px solid #d8d0bd; border-top:4px solid var(--navy); background:linear-gradient(180deg,#fffaf0 0%,#f8f1e2 100%); box-shadow:0 18px 50px -30px rgba(26,40,73,.45); margin-bottom:28px; position:relative;}
.hero:before{content:'';position:absolute;top:-4px;left:-1px;right:-1px;height:4px;background:linear-gradient(90deg,var(--navy),var(--gold),var(--coral));}
.eyebrow{font-family:'Inter',sans-serif; font-size:11px; letter-spacing:.22em; text-transform:uppercase; color:var(--gold); font-weight:700; margin-bottom:12px;}
.hero h1{font-family:'Cormorant Garamond',serif; color:var(--navy)!important; font-size:56px; line-height:1.02; font-weight:500; letter-spacing:-.025em; margin:0 0 12px;}
.hero p{font-size:19px; font-style:italic; color:var(--muted); line-height:1.55; max-width:950px; margin:0;}
.badge{display:inline-flex; padding:7px 12px; margin:14px 6px 0 0; border-radius:0; background:rgba(184,137,63,.10); border:1px solid rgba(184,137,63,.28); color:var(--navy); font-family:'Inter',sans-serif; font-size:12px; font-weight:700; letter-spacing:.03em;}
.section-title{font-family:'Inter',sans-serif; font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--navy); font-weight:800; border-bottom:2px solid var(--gold); padding-bottom:5px; display:inline-block; margin:24px 0 14px;}
.card{background:white; border:1px solid #e0d8c0; padding:22px 24px; box-shadow:0 12px 32px -22px rgba(26,40,73,.25); height:100%;}
.kpi{background:var(--bone); border-top:3px solid var(--gold); padding:18px 18px 15px; min-height:118px;}
.kpi .label{font-family:'Inter',sans-serif; font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:800;}
.kpi .value{font-family:'Cormorant Garamond',serif; color:var(--navy); font-size:34px; line-height:1.05; margin-top:6px; font-weight:600;}
.kpi .note{font-size:13px; color:var(--muted); margin-top:7px; line-height:1.35;}
.policy{background:white; border:1px solid #d8d0bd; padding:28px 32px; margin:16px 0 24px; position:relative;}
.policy:before{content:''; position:absolute; top:-1px; left:-1px; right:-1px; height:4px; background:linear-gradient(90deg,var(--navy),var(--gold),var(--coral));}
.policy h3{font-family:'Cormorant Garamond',serif; color:var(--navy)!important; font-size:28px; margin:0 0 12px; font-weight:500;}
.policy li{margin-bottom:8px; font-size:16px; line-height:1.65;}
.formula{font-family:'Cormorant Garamond',serif; font-size:21px; text-align:center; color:var(--navy); background:linear-gradient(180deg,#fdfaf3 0%,#f8f3e6 100%); border-left:3px solid var(--navy); padding:18px 24px; margin:12px 0;}
.stTabs [data-baseweb="tab-list"]{gap:8px; border-bottom:1px solid #d8d0bd;}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif; background:#f5f1e8; border:1px solid #e0d8c0; padding:10px 16px; color:var(--navy);}
.stTabs [aria-selected="true"]{background:var(--navy)!important; color:var(--bone)!important;}
[data-testid="stMetricValue"]{font-family:'Cormorant Garamond',serif; color:var(--navy);}
.stDataFrame{border:1px solid #e0d8c0;}
hr{border-color:#d8d0bd;}
h1,h2,h3{color:var(--navy)!important;}
.small-note{color:var(--muted);font-style:italic;font-size:14px;}

.pretty-table{width:100%; border-collapse:separate; border-spacing:0 10px; font-family:'Inter',sans-serif;}
.pretty-table th{font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:#1a2849; text-align:left; padding:10px 14px; border-bottom:2px solid #b8893f;}
.pretty-table td{background:#fffdf7; border-top:1px solid #e0d8c0; border-bottom:1px solid #e0d8c0; padding:14px 14px; vertical-align:top; color:#1a1a1a;}
.pretty-table td:first-child{border-left:4px solid #b8893f; font-family:'Cormorant Garamond',serif; font-size:24px; color:#1a2849; text-align:center; width:60px;}
.pretty-table td:last-child{border-right:1px solid #e0d8c0;}
.roadmap-card{background:#fffdf7;border:1px solid #e0d8c0;border-left:4px solid #b8893f;padding:18px 20px;margin:10px 0;box-shadow:0 12px 30px -24px rgba(26,40,73,.25)}
.roadmap-title{font-family:'Cormorant Garamond',serif;font-size:25px;color:#1a2849;margin-bottom:6px}.roadmap-meta{font-family:'Inter',sans-serif;font-size:13px;color:#6b6b6b;line-height:1.55}.roadmap-num{font-family:'Inter',sans-serif;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#b8893f;font-weight:800}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def macro_data():
    return pd.DataFrame({
        "Năm":[2020,2021,2022,2023,2024,2025],
        "GDP":[8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
        "K":[16500,17800,19600,21300,23500,25900],
        "L":[53.6,50.5,51.7,52.4,52.9,53.4],
        "D":[12.0,12.7,14.3,16.5,18.3,19.5],
        "AI":[55.6,60.2,65.4,67.0,73.8,80.1],
        "H":[24.1,26.1,26.2,27.0,28.4,29.2]
    })

@st.cache_data
def sectors_data():
    return pd.DataFrame({
        "Ngành":["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Khai khoáng","Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo","Y tế"],
        "Tăng trưởng (%)":[3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
        "Năng suất":[103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1],
        "Lan tỏa":[0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
        "Xuất khẩu":[40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
        "Việc làm":[13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
        "AI readiness":[15,55,20,30,48,72,42,88,38,45],
        "Rủi ro TĐH":[18,42,25,55,38,52,35,28,22,18]
    })

@st.cache_data
def regions_data():
    return pd.DataFrame({
        "Vùng":["Trung du miền núi phía Bắc","Đồng bằng sông Hồng","Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","Đồng bằng sông Cửu Long"],
        "Mã":["NMM","RRD","NCC","CH","SE","MD"],
        "Digital Index":[38,78,55,32,82,48],
        "GRDP/người":[57.0,152.3,87.5,68.9,158.9,80.5],
        "FDI":[3.5,20.0,8.2,0.8,18.5,2.1],
        "AI readiness":[22,68,40,18,75,30],
        "LĐ đào tạo":[21.5,36.8,27.5,18.2,42.5,16.8],
        "R&D/GRDP":[0.18,0.85,0.32,0.15,0.78,0.22],
        "Internet":[72,92,84,68,94,78],
        "Gini":[0.405,0.358,0.372,0.412,0.385,0.392]
    })

@st.cache_data
def projects_data():
    return pd.DataFrame({
        "Mã":[f"P{i}" for i in range(1,16)],
        "Tên dự án":["Trung tâm dữ liệu quốc gia Hòa Lạc","Trung tâm dữ liệu quốc gia phía Nam","Hệ thống 5G phủ sóng toàn quốc","Hệ thống định danh điện tử VNeID 2.0","Cổng dịch vụ công quốc gia v3","Y tế số quốc gia","Giáo dục số K-12 toàn quốc","Trung tâm AI quốc gia + supercomputing","Sandbox tài chính số","Logistics thông minh + cảng biển số","Nông nghiệp số ĐBSCL","Đào tạo 50.000 kỹ sư AI/bán dẫn","Khu CN bán dẫn Bắc Ninh - Bắc Giang","An ninh mạng quốc gia SOC","Open Data + dữ liệu mở quốc gia"],
        "Lĩnh vực":["Hạ tầng","Hạ tầng","Hạ tầng","Chính phủ số","Chính phủ số","Y tế số","Giáo dục","AI","Tài chính số","Logistics","Nông nghiệp","Nhân lực","Bán dẫn","An ninh","Dữ liệu"],
        "Chi phí":[12000,11500,18000,4500,3200,5800,6500,15000,2500,7200,4800,8500,20000,3800,1500],
        "Lợi ích NPV":[21500,20800,32500,9200,6800,11400,12200,28500,5800,13800,8500,16200,35000,7500,3800],
        "Năm 1-2":[8500,7500,12000,3500,2500,4000,4500,9000,1800,5000,3500,5500,13000,2800,1200],
        "Năm 3-5":[3500,4000,6000,1000,700,1800,2000,6000,700,2200,1300,3000,7000,1000,300]
    })

# =========================================================
# HELPERS
# =========================================================
def hero(title, subtitle, badges=()):
    badge_html = "".join([f"<span class='badge'>{b}</span>" for b in badges])
    st.markdown(f"""
    <div class='hero'>
      <div class='eyebrow'>AIDEOM-VN · DECISION OPTIMIZATION</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

def kpi(label, value, note=""):
    st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>", unsafe_allow_html=True)

def section(text):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)

def clean_fig(fig, height=440):
    fig.update_layout(
        height=height, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.75)",
        margin=dict(l=25,r=20,t=55,b=25), font=dict(family="Inter", color="#1a1a1a"),
        title_font=dict(family="Cormorant Garamond", size=26, color="#1a2849"),
        legend=dict(bgcolor="rgba(255,255,255,.65)", bordercolor="#e0d8c0", borderwidth=1)
    )
    return fig

def norm_good(s):
    s = pd.Series(s, dtype=float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s*0

def norm_bad(s):
    s = pd.Series(s, dtype=float)
    return (s.max() - s) / (s.max() - s.min()) if s.max() != s.min() else s*0

def policy_box(title, bullets):
    li = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(f"<div class='policy'><h3>{title}</h3><ul>{li}</ul></div>", unsafe_allow_html=True)

# =========================================================
# SOLVERS FOR BAI 3, 4, 5
# =========================================================
def compute_priority(df, weights):
    # weights keys: growth, productivity, spillover, export, employment, ai, risk
    cols = ["Tăng trưởng (%)","Năng suất","Lan tỏa","Xuất khẩu","Việc làm","AI readiness"]
    X = pd.DataFrame({c: norm_good(df[c]) for c in cols})
    X["Giảm rủi ro"] = norm_bad(df["Rủi ro TĐH"])
    w = np.array([weights["growth"],weights["productivity"],weights["spillover"],weights["export"],weights["employment"],weights["ai"],weights["risk"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(7)/7
    score = X.values @ w
    out = pd.concat([df[["Ngành"]].copy(), X], axis=1)
    out["Priority"] = score
    out["Xếp hạng"] = out["Priority"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("Priority", ascending=False), w

def solve_region_lp(B=50000, floor=5000, cap=12000, H_floor=12000, gamma=0.002, lam=0.7, fairness=True, soft_fallback=True):
    """Solve Bài 4 robustly.
    Hard C5 from the assignment can be infeasible at gamma=0.002, lambda=0.7, cap=12000
    because low-digital regions cannot catch up enough within their regional cap.
    This function first tries hard fairness; if infeasible, it automatically solves a soft-fairness LP
    with slack variables and reports the required fairness gap instead of crashing.
    """
    regs = ["NMM","RRD","NCC","CH","SE","MD"]
    beta = np.array([
        [1.15,0.85,0.55,1.30],
        [0.95,1.25,1.40,1.05],
        [1.05,0.95,0.85,1.15],
        [1.20,0.75,0.45,1.35],
        [0.90,1.30,1.55,1.00],
        [1.10,0.85,0.65,1.25]
    ], dtype=float)
    D0 = np.array([38,78,55,32,82,48], dtype=float)

    def build_and_solve(mode="hard"):
        # vars: 24 allocation variables + M + optional 6 slack variables for soft C5
        add_M = fairness
        add_slack = fairness and mode == "soft"
        n = 24 + (1 if add_M else 0) + (6 if add_slack else 0)
        M_idx = 24 if add_M else None
        S_idx = 25 if add_slack else None
        c = np.zeros(n)
        c[:24] = -beta.flatten()
        if add_slack:
            # penalty chosen large enough to minimize fairness violation while still returning a feasible plan
            c[S_idx:S_idx+6] = 1000.0
        A_ub=[]; b_ub=[]
        # total budget
        row=np.zeros(n); row[:24]=1; A_ub.append(row); b_ub.append(B)
        # region cap/floor
        for r in range(6):
            row=np.zeros(n); row[r*4:(r+1)*4]=1; A_ub.append(row); b_ub.append(cap)
            row=np.zeros(n); row[r*4:(r+1)*4]=-1; A_ub.append(row); b_ub.append(-floor)
        # human-capital floor
        row=np.zeros(n)
        for r in range(6): row[r*4+3] = -1
        A_ub.append(row); b_ub.append(-H_floor)
        if fairness:
            for r in range(6):
                # D0_r + gamma*x_Dr <= M  -> gamma*x_Dr - M <= -D0_r
                row=np.zeros(n); row[r*4+1]=gamma; row[M_idx]=-1
                A_ub.append(row); b_ub.append(-D0[r])
            for r in range(6):
                # hard: D0_r + gamma*x_Dr >= lam*M
                # soft: D0_r + gamma*x_Dr + s_r >= lam*M
                # -> -gamma*x_Dr + lam*M - s_r <= D0_r
                row=np.zeros(n); row[r*4+1]=-gamma; row[M_idx]=lam
                if add_slack: row[S_idx+r] = -1
                A_ub.append(row); b_ub.append(D0[r])
        bounds=[(0,None)]*n
        return linprog(c,A_ub=np.array(A_ub),b_ub=np.array(b_ub),bounds=bounds,method="highs")

    res = build_and_solve("hard")
    mode = "hard"
    if fairness and (not res.success) and soft_fallback:
        res = build_and_solve("soft")
        mode = "soft"
    if not res.success:
        return None, res.message

    x=res.x[:24].reshape(6,4)
    alloc=pd.DataFrame(x, index=["Trung du miền núi phía Bắc","Đồng bằng sông Hồng","Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","Đồng bằng sông Cửu Long"], columns=["Hạ tầng số I","CĐS DN D","AI","Nhân lực H"])
    raw_obj = (beta*x).sum()
    post=D0 + gamma*x[:,1]
    slack = None
    if fairness and mode == "soft":
        slack = res.x[25:31]
    return {"alloc":alloc,"Z":raw_obj,"post_digital":post,"beta":beta,"D0":D0,"M":res.x[24] if fairness else None,"mode":mode,"slack":slack}, None

def solve_projects(budget=80000, y12_budget=40000, force_both_dc=False, force_p15=False, min_projects=7, max_projects=11, risk_adjust=False):
    df=projects_data()
    C=df["Chi phí"].to_numpy(); C1=df["Năm 1-2"].to_numpy(); B=df["Lợi ích NPV"].to_numpy(dtype=float)
    if risk_adjust:
        p=[]
        for f in df["Lĩnh vực"]:
            if f == "Hạ tầng": p.append(0.85)
            elif f == "Chính phủ số": p.append(0.75)
            elif f in ["AI","Bán dẫn"]: p.append(0.65)
            else: p.append(0.80)
        B = B*np.array(p)
    best=None
    feasible_count=0
    for bits in itertools.product([0,1], repeat=15):
        y=np.array(bits)
        n=y.sum()
        if n < min_projects or n > max_projects: continue
        if (C*y).sum() > budget or (C1*y).sum() > y12_budget: continue
        if force_both_dc:
            if y[0] != 1 or y[1] != 1: continue
        else:
            if y[0] + y[1] > 1: continue
        if y[7] > y[11]: continue      # P8 <= P12
        if y[12] > y[11]: continue     # P13 <= P12
        if y[3] + y[4] < 1: continue   # digital gov
        if y[13] < 1: continue         # P14 mandatory
        if force_p15 and y[14] < 1: continue
        feasible_count += 1
        val=(B*y).sum()
        if best is None or val > best[0]: best=(val,y)
    if best is None:
        return None, feasible_count
    val,y=best
    out=df.copy()
    out["Chọn"] = np.where(y==1,"Có","Không")
    out["Tỷ suất NPV/Chi phí"] = out["Lợi ích NPV"] / out["Chi phí"]
    return {"value":val,"y":y,"table":out,"cost":(C*y).sum(),"y12":(C1*y).sum(),"count":int(y.sum()),"feasible_count":feasible_count}, feasible_count

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("# AIDEOM<span style='color:#e9c46a'>·</span>VN", unsafe_allow_html=True)
st.sidebar.caption("Premium Streamlit Dashboard · 12 bài")
page = st.sidebar.radio("Điều hướng", [
    "Trang chủ", "Bài 1 · Cobb-Douglas", "Bài 2 · LP ngân sách", "Bài 3 · Priority ngành",
    "Bài 4 · LP ngành-vùng", "Bài 5 · MIP dự án", "Bài 6 · TOPSIS vùng", "Bài 7 · Pareto đa mục tiêu",
    "Bài 8 · Tối ưu động", "Bài 9 · AI & lao động", "Bài 10 · Stochastic LP", "Bài 11 · Q-learning", "Bài 12 · Tích hợp"
])
st.sidebar.markdown("---")
st.sidebar.caption("Bản V4: giao diện sáng, bỏ bảng đen, sửa cứng Bài 4 bằng hard/soft fairness fallback.")

# =========================================================
# HOME
# =========================================================
if page == "Trang chủ":
    hero("Phát triển kinh tế Việt Nam trong kỉ nguyên AI", "Dashboard Streamlit tương tác cho 12 bài AIDEOM-VN: bảng có thể chỉnh, biểu đồ hover/zoom, tham số kịch bản, và phần nhận xét chính sách đầy đủ.", ["Editorial UI", "Interactive tables", "Policy notes", "Bài 1–12"])
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Phạm vi", "12 bài", "Dễ → rất khó")
    with c2: kpi("Dữ liệu", "2020–2025", "macro · sector · region")
    with c3: kpi("Kỹ thuật", "7 nhóm", "LP/MIP/TOPSIS/Pareto/DP/SP/RL")
    with c4: kpi("Bản này", "V4", "light UI · fix bài 4")
    section("Mục lục tương tác")
    roadmap = pd.DataFrame({
        "Bài":[1,2,3,4,5,6,7,8,9,10,11,12],
        "Tên":["Cobb-Douglas mở rộng","LP phân bổ ngân sách số","Priority 10 ngành","LP ngành × vùng","MIP chọn dự án","TOPSIS 6 vùng","Pareto đa mục tiêu","Tối ưu động 2026–2035","AI và lao động","Quy hoạch ngẫu nhiên","Q-learning","Dashboard tích hợp"],
        "Tương tác chính":["hệ số co giãn, kịch bản 2030","ngân sách & sàn đầu tư","trọng số chính sách, heatmap","công bằng vùng, heatmap phân bổ","ngân sách, ràng buộc dự án","trọng số chuyên gia/entropy","lọc nghiệm Pareto","chiết khấu & ngân sách năm","AI adoption & đào tạo lại","xác suất kịch bản","episode/alpha/gamma","5 kịch bản chính sách"],
        "Đầu ra":["TFP, MAPE, GDP 2030","phân bổ tối ưu, Z*","xếp hạng ngành","ma trận 6×4, chi phí công bằng","danh mục dự án","xếp hạng vùng","frontier & compromise","policy heatmap","việc làm ròng","SP/EVPI/VSS","learning curve","bảng tổng hợp"]
    })
    cards = "".join([f"<div class='roadmap-card'><div class='roadmap-num'>Bài {int(r['Bài']):02d}</div><div class='roadmap-title'>{r['Tên']}</div><div class='roadmap-meta'><b>Tương tác:</b> {r['Tương tác chính']}<br><b>Đầu ra:</b> {r['Đầu ra']}</div></div>" for _, r in roadmap.iterrows()])
    st.markdown(cards, unsafe_allow_html=True)
    policy_box("Định hướng thiết kế", [
        "Giao diện mô phỏng phong cách web học thuật: nền giấy, sidebar xanh navy, tiêu đề serif, thẻ kết quả và khối phân tích chính sách.",
        "Mỗi bài có ít nhất một dạng tương tác: slider, data editor, biểu đồ Plotly, heatmap hoặc bộ lọc kịch bản.",
        "Các nhận xét chính sách được giữ lại để app không chỉ là demo kỹ thuật mà còn là sản phẩm phân tích ra quyết định."
    ])

# =========================================================
# BAI 1
# =========================================================
elif page == "Bài 1 · Cobb-Douglas":
    hero("Bài 1. Hàm sản xuất Cobb-Douglas mở rộng", "Ước lượng TFP, kiểm tra sai số dự báo và mô phỏng GDP Việt Nam năm 2030.", ["TFP", "Growth accounting", "2030 scenario"])
    df=macro_data()
    with st.sidebar:
        st.subheader("Hệ số co giãn")
        a=st.slider("α vốn K",0.05,0.60,0.33,0.01); b=st.slider("β lao động L",0.05,0.70,0.42,0.01)
        g=st.slider("γ số hóa D",0.00,0.30,0.10,0.01); d=st.slider("δ AI",0.00,0.25,0.08,0.01); t=st.slider("θ nhân lực số",0.00,0.25,0.07,0.01)
        normalize=st.checkbox("Chuẩn hóa tổng hệ số = 1", True)
        st.subheader("Kịch bản 2030")
        D30=st.slider("D 2030 (%)",20.0,40.0,30.0,0.5); AI30=st.slider("AI 2030 (nghìn DN)",80.0,160.0,100.0,1.0); H30=st.slider("H 2030 (%)",28.0,50.0,35.0,0.5)
    coeff=np.array([a,b,g,d,t])
    if normalize: coeff=coeff/coeff.sum()
    a,b,g,d,t=coeff
    Y,K,L,D,AI,H=[df[c].to_numpy(float) for c in ["GDP","K","L","D","AI","H"]]
    A=Y/(K**a*L**b*D**g*AI**d*H**t); Abar=A.mean(); Yhat=Abar*(K**a*L**b*D**g*AI**d*H**t)
    mape=np.mean(np.abs((Y-Yhat)/Y))*100
    Y30=A[-1]*(1.012**5)*(K[-1]*1.06**5)**a*(L[-1]*1.006**5)**b*(D30**g)*(AI30**d)*(H30**t)
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi("TFP 2025", f"{A[-1]:.2f}", f"2020: {A[0]:.2f}")
    with c2: kpi("MAPE", f"{mape:.2f}%", "A trung bình")
    with c3: kpi("GDP 2030", f"{Y30:,.0f}", "nghìn tỷ VND")
    with c4: kpi("Tổng hệ số", f"{coeff.sum():.2f}", "lợi suất quy mô")
    tab1,tab2,tab3=st.tabs(["📈 Biểu đồ", "🧮 Phân rã", "📝 Nhận xét"])
    with tab1:
        plot=df[["Năm","GDP"]].copy(); plot["GDP dự báo"]=Yhat; plot["TFP"]=A
        st.plotly_chart(clean_fig(px.line(plot,x="Năm",y=["GDP","GDP dự báo"],markers=True,title="GDP thực tế và GDP dự báo")), use_container_width=True)
        st.plotly_chart(clean_fig(px.line(plot,x="Năm",y="TFP",markers=True,title="TFP A_t theo thời gian")), use_container_width=True)
    with tab2:
        dln=lambda x: np.log(x[-1]/x[0])/5
        comps={"TFP":dln(A),"K":a*dln(K),"L":b*dln(L),"D":g*dln(D),"AI":d*dln(AI),"H":t*dln(H)}
        total=sum(comps.values())
        compdf=pd.DataFrame({"Yếu tố":list(comps),"Đóng góp %":[v/total*100 for v in comps.values()]})
        st.dataframe(compdf, use_container_width=True, hide_index=True)
        st.plotly_chart(clean_fig(px.bar(compdf,x="Yếu tố",y="Đóng góp %",text="Đóng góp %",title="Cơ cấu đóng góp tăng trưởng")),use_container_width=True)
    with tab3:
        policy_box("Nhận xét và hàm ý chính sách", [
            "TFP tăng cho thấy tăng trưởng không chỉ đến từ tích lũy vốn mà còn từ hiệu quả, công nghệ và năng lực quản trị.",
            "Nếu D, AI và H không tăng đồng bộ, mục tiêu GDP 2030 dễ phụ thuộc quá mức vào K, làm giảm chất lượng tăng trưởng.",
            "Chính sách nên ưu tiên năng lực hấp thụ công nghệ: đào tạo kỹ năng số, dữ liệu mở, hạ tầng tính toán và cơ chế thử nghiệm AI."
        ])

# =========================================================
# BAI 2
# =========================================================
elif page == "Bài 2 · LP ngân sách":
    hero("Bài 2. Quy hoạch tuyến tính phân bổ ngân sách", "Tối đa hóa tăng GDP kỳ vọng khi phân bổ ngân sách cho hạ tầng, AI, nhân lực và R&D.", ["Linear Programming", "Shadow price", "Sensitivity"])
    with st.sidebar:
        B=st.slider("Ngân sách tổng",70,160,100,5)
        minI=st.slider("Sàn hạ tầng số",0,50,25); minAI=st.slider("Sàn AI",0,50,15); minH=st.slider("Sàn nhân lực",0,50,20); minR=st.slider("Sàn R&D",0,50,10)
        strategic=st.slider("Tỷ trọng AI + R&D tối thiểu",0.0,0.70,0.35,0.05)
    c=np.array([-0.85,-1.20,-0.95,-1.35])
    A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[strategic, -(1-strategic), strategic, -(1-strategic)]]
    b_ub=[B,-minI,-minAI,-minH,-minR,0]
    res=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method="highs")
    if not res.success:
        st.error("Bài toán không khả thi với ràng buộc hiện tại.")
    else:
        x=res.x; Z=-res.fun
        c1,c2,c3=st.columns(3)
        with c1: kpi("Z*", f"{Z:.2f}", "nghìn tỷ VND GDP")
        with c2: kpi("AI + R&D", f"{(x[1]+x[3])/x.sum()*100:.1f}%", "tỷ trọng chiến lược")
        with c3: kpi("Sử dụng", f"{x.sum():.0f}/{B}", "nghìn tỷ VND")
        alloc=pd.DataFrame({"Hạng mục":["Hạ tầng số","AI & dữ liệu","Nhân lực số","R&D"],"Phân bổ":x,"Hệ số":[0.85,1.20,0.95,1.35]})
        tab1,tab2=st.tabs(["📊 Kết quả", "📝 Chính sách"])
        with tab1:
            st.dataframe(alloc,use_container_width=True,hide_index=True)
            st.plotly_chart(clean_fig(px.bar(alloc,x="Hạng mục",y="Phân bổ",text="Phân bổ",title="Phân bổ ngân sách tối ưu")),use_container_width=True)
            vals=[]
            for bgt in range(80,161,10):
                rr=linprog(c,A_ub=A_ub,b_ub=[bgt,-minI,-minAI,-minH,-minR,0],bounds=[(0,None)]*4,method="highs")
                vals.append({"B":bgt,"Z*":-rr.fun if rr.success else np.nan})
            st.plotly_chart(clean_fig(px.line(pd.DataFrame(vals),x="B",y="Z*",markers=True,title="Đường cong độ nhạy Z*(B)")),use_container_width=True)
        with tab2:
            policy_box("Nhận xét và hàm ý chính sách", [
                "R&D thường được phân bổ mạnh vì hệ số tác động biên cao nhất, nhưng cần đi kèm năng lực hấp thụ và cơ chế đánh giá kết quả.",
                "Ràng buộc tỷ trọng AI + R&D giúp tránh kịch bản toàn bộ ngân sách bị hút vào hạ tầng truyền thống.",
                "Shadow price của ngân sách tổng có thể hiểu là lợi ích biên khi tăng thêm 1 đơn vị ngân sách, nhưng chỉ đúng trong vùng ràng buộc hiện tại."
            ])

# =========================================================
# BAI 3 — FIXED
# =========================================================
elif page == "Bài 3 · Priority ngành":
    hero("Bài 3. Chỉ số ưu tiên Priority cho 10 ngành", "Chuẩn hóa min-max, chỉnh trọng số chính sách và xếp hạng ngành nên ưu tiên chuyển đổi số & AI.", ["Fixed", "Data editor", "Weight sliders", "Heatmap"])
    st.sidebar.subheader("Trọng số chính sách")
    weights={
        "growth":st.sidebar.slider("Tăng trưởng",0.0,0.5,0.15,0.01),
        "productivity":st.sidebar.slider("Năng suất",0.0,0.5,0.15,0.01),
        "spillover":st.sidebar.slider("Lan tỏa",0.0,0.5,0.20,0.01),
        "export":st.sidebar.slider("Xuất khẩu",0.0,0.5,0.15,0.01),
        "employment":st.sidebar.slider("Việc làm",0.0,0.5,0.10,0.01),
        "ai":st.sidebar.slider("AI readiness",0.0,0.6,0.20,0.01),
        "risk":st.sidebar.slider("Giảm rủi ro TĐH",0.0,0.5,0.15,0.01),
    }
    st.sidebar.caption("Rủi ro được đảo chiều thành tiêu chí tốt: rủi ro thấp → điểm cao. Bản V3 đã sửa lỗi dấu ở phần này.")
    section("Bảng dữ liệu ngành có thể chỉnh")
    raw=st.data_editor(sectors_data(), use_container_width=True, hide_index=True, num_rows="fixed")
    ranking,w_norm=compute_priority(raw,weights)
    c1,c2,c3=st.columns(3)
    top=ranking.iloc[0]
    with c1: kpi("Top 1", top["Ngành"], f"Priority {top['Priority']:.3f}")
    with c2: kpi("Trọng số AI", f"{w_norm[5]*100:.1f}%", "sau chuẩn hóa")
    with c3: kpi("Trọng số giảm rủi ro", f"{w_norm[6]*100:.1f}%", "sau chuẩn hóa")
    tab1,tab2,tab3,tab4=st.tabs(["🏆 Xếp hạng", "🔥 Heatmap", "🧪 Độ nhạy AI", "📝 Nhận xét"])
    with tab1:
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.plotly_chart(clean_fig(px.bar(ranking.sort_values("Priority"),x="Priority",y="Ngành",orientation="h",text="Priority",title="Xếp hạng Priority theo ngành"),520), use_container_width=True)
    with tab2:
        heat=ranking.set_index("Ngành")[["Tăng trưởng (%)","Năng suất","Lan tỏa","Xuất khẩu","Việc làm","AI readiness","Giảm rủi ro","Priority"]]
        fig=px.imshow(heat, aspect="auto", text_auto=".2f", title="Ma trận chuẩn hóa và điểm Priority")
        st.plotly_chart(clean_fig(fig,560), use_container_width=True)
    with tab3:
        rows=[]
        for wai in np.arange(0.05,0.41,0.05):
            ww=weights.copy(); ww["ai"]=wai
            rank_i,_=compute_priority(raw,ww)
            for _,r in rank_i.head(3).iterrows(): rows.append({"w_AI":round(wai,2),"Ngành":r["Ngành"],"Rank":int(r["Xếp hạng"]),"Priority":r["Priority"]})
        sens=pd.DataFrame(rows)
        st.dataframe(sens,use_container_width=True,hide_index=True)
        st.plotly_chart(clean_fig(px.line(sens,x="w_AI",y="Priority",color="Ngành",markers=True,title="Độ nhạy top-3 khi thay đổi trọng số AI")),use_container_width=True)
    with tab4:
        top3 = ", ".join(ranking.head(3)["Ngành"].tolist())
        policy_box("Nhận xét và hàm ý chính sách", [
            f"Với trọng số hiện tại, nhóm ưu tiên cao nhất là: <b>{top3}</b>. Đây là các ngành có kết hợp tốt giữa lan tỏa, xuất khẩu, năng lực AI và quy mô tác động.",
            "Khai khoáng có năng suất rất cao nhưng điểm Priority không nhất thiết cao vì tăng trưởng thấp, rủi ro tự động hóa cao và hiệu ứng lan tỏa số thấp hơn các ngành nền tảng.",
            "Bộ trọng số không nên do riêng kỹ thuật quyết định; cần hội đồng chính sách, chuyên gia ngành và đối thoại công khai để bảo đảm tính chính danh.",
            "Khi tăng trọng số AI readiness, CNTT-Truyền thông và Tài chính-Ngân hàng thường vươn lên; khi tăng trọng số việc làm/bao trùm, nông nghiệp, bán lẻ và giáo dục có thể được ưu tiên hơn."
        ])

# =========================================================
# BAI 4 — FIXED
# =========================================================
elif page == "Bài 4 · LP ngành-vùng":
    hero("Bài 4. LP phân bổ ngân sách số theo vùng", "Giải LP 6 vùng × 4 hạng mục với ràng buộc công bằng vùng miền, sàn/trần ngân sách và sàn nhân lực số.", ["Fixed", "24 variables", "Fairness constraint", "Heatmap"])
    with st.sidebar:
        B=st.slider("Ngân sách tổng",30000,70000,50000,1000)
        floor=st.slider("Sàn mỗi vùng",0,10000,5000,500)
        cap=st.slider("Trần mỗi vùng",6000,20000,12000,500)
        H_floor=st.slider("Sàn nhân lực số H",0,25000,12000,500)
        gamma=st.slider("γ cải thiện số hóa",0.0005,0.0050,0.0020,0.0001,format="%.4f")
        lam=st.slider("λ công bằng",0.30,0.95,0.70,0.05)
        fairness=st.checkbox("Bật ràng buộc công bằng C5", True)
    sol,err=solve_region_lp(B,floor,cap,H_floor,gamma,lam,fairness)
    if sol is None:
        st.error(f"Bài toán không khả thi: {err}")
        st.info("Gợi ý: giảm sàn mỗi vùng/H hoặc tăng ngân sách/trần mỗi vùng.")
    else:
        if sol.get("mode") == "soft":
            st.warning("Ràng buộc công bằng C5 bản gốc đang không khả thi với bộ tham số hiện tại. App tự chuyển sang **soft fairness**: vẫn tối ưu được phân bổ và báo phần thiếu công bằng để phân tích chính sách, thay vì làm dashboard bị lỗi.")
        alloc=sol["alloc"]
        nofair,_=solve_region_lp(B,floor,cap,H_floor,gamma,lam,False)
        cost_fair=(nofair["Z"]-sol["Z"]) if (fairness and nofair) else 0
        c1,c2,c3,c4=st.columns(4)
        with c1: kpi("Z*", f"{sol['Z']:,.0f}", "GDP gain kỳ vọng")
        with c2: kpi("Vùng cao nhất", alloc.sum(axis=1).idxmax(), f"{alloc.sum(axis=1).max():,.0f}")
        with c3: kpi("Hạng mục lớn nhất", alloc.sum(axis=0).idxmax(), f"{alloc.sum(axis=0).max():,.0f}")
        with c4: kpi("Chi phí công bằng", f"{cost_fair:,.0f}", "so với không C5")
        tab1,tab2,tab3,tab4=st.tabs(["📊 Ma trận phân bổ", "🔥 Heatmap", "⚖️ Công bằng", "📝 Nhận xét"])
        with tab1:
            st.dataframe(alloc.style.format("{:,.0f}"), use_container_width=True)
            long=alloc.reset_index(names="Vùng").melt(id_vars="Vùng",var_name="Hạng mục",value_name="Ngân sách")
            st.plotly_chart(clean_fig(px.bar(long,x="Vùng",y="Ngân sách",color="Hạng mục",title="Cơ cấu phân bổ theo vùng",barmode="stack"),520), use_container_width=True)
        with tab2:
            st.plotly_chart(clean_fig(px.imshow(alloc, text_auto=".0f", aspect="auto", title="Heatmap ngân sách 6 vùng × 4 hạng mục"),520), use_container_width=True)
        with tab3:
            fdf=pd.DataFrame({"Vùng":alloc.index,"Digital ban đầu":sol["D0"],"Digital sau đầu tư":sol["post_digital"],"Tổng ngân sách":alloc.sum(axis=1).values})
            if sol.get("slack") is not None:
                fdf["Thiếu so với C5"] = sol["slack"]
            st.dataframe(fdf,use_container_width=True,hide_index=True)
            st.plotly_chart(clean_fig(px.line(fdf,x="Vùng",y=["Digital ban đầu","Digital sau đầu tư"],markers=True,title="Digital Index trước và sau phân bổ")),use_container_width=True)
        with tab4:
            policy_box("Nhận xét và hàm ý chính sách", [
                "Bật C5 làm mô hình không chỉ tối đa hóa GDP gain mà còn bảo đảm khoảng cách số giữa vùng yếu và vùng mạnh không vượt ngưỡng chính sách.",
                "Nếu bỏ công bằng, vốn thường chảy về vùng có hệ số AI/D cao như Đông Nam Bộ và Đồng bằng sông Hồng; điều này tối ưu ngắn hạn nhưng có thể làm phân hóa vùng miền.",
                "Tây Nguyên và miền núi phía Bắc có AI readiness thấp nên nên ưu tiên hạ tầng số và nhân lực trước khi đẩy mạnh AI chuyên sâu.",
                "Chi phí công bằng là phần Z* giảm khi giữ C5. Đây là con số nên được trình bày như chi phí xã hội chấp nhận để đạt phát triển bao trùm."
            ])

# =========================================================
# BAI 5 — FIXED
# =========================================================
elif page == "Bài 5 · MIP dự án":
    hero("Bài 5. Quy hoạch nguyên hỗn hợp chọn dự án", "Chọn tập dự án chuyển đổi số tối ưu với ràng buộc ngân sách, loại trừ, tiên quyết, bắt buộc và số lượng dự án.", ["Fixed", "Binary selection", "No PuLP needed", "Scenario"])
    with st.sidebar:
        budget=st.slider("Ngân sách 5 năm",50000,120000,80000,2500)
        y12_budget=st.slider("Ngân sách năm 1–2",25000,70000,40000,2500)
        force_both_dc=st.checkbox("Bắt buộc cả P1 và P2", False)
        force_p15=st.checkbox("Bắt buộc P15 Open Data", False)
        risk_adjust=st.checkbox("Tối đa hóa lợi ích kỳ vọng theo rủi ro", False)
        minp=st.slider("Số dự án tối thiểu",1,15,7)
        maxp=st.slider("Số dự án tối đa",1,15,11)
    sol,feas=solve_projects(budget,y12_budget,force_both_dc,force_p15,minp,maxp,risk_adjust)
    if sol is None:
        st.error("Không tìm được phương án khả thi với bộ ràng buộc hiện tại.")
        st.info("Gợi ý: tăng ngân sách, tăng ngân sách năm 1–2, hoặc nới số dự án tối thiểu/tối đa.")
    else:
        selected=sol["table"][sol["table"]["Chọn"]=="Có"].copy()
        c1,c2,c3,c4=st.columns(4)
        with c1: kpi("Z*", f"{sol['value']:,.0f}", "NPV / kỳ vọng")
        with c2: kpi("Tổng chi phí", f"{sol['cost']:,.0f}", f"trần {budget:,.0f}")
        with c3: kpi("Năm 1–2", f"{sol['y12']:,.0f}", f"trần {y12_budget:,.0f}")
        with c4: kpi("Số dự án", f"{sol['count']}", f"khả thi: {sol['feasible_count']}")
        tab1,tab2,tab3,tab4=st.tabs(["✅ Dự án chọn", "📋 Toàn bộ dự án", "🧪 Kịch bản", "📝 Nhận xét"])
        with tab1:
            st.dataframe(selected,use_container_width=True,hide_index=True)
            st.plotly_chart(clean_fig(px.bar(selected,x="Mã",y=["Chi phí","Lợi ích NPV"],barmode="group",hover_data=["Tên dự án","Lĩnh vực"],title="Chi phí và lợi ích của các dự án được chọn")),use_container_width=True)
            field=selected.groupby("Lĩnh vực",as_index=False)[["Chi phí","Lợi ích NPV"]].sum()
            st.plotly_chart(clean_fig(px.pie(field,names="Lĩnh vực",values="Chi phí",title="Cơ cấu chi phí theo lĩnh vực"),420),use_container_width=True)
        with tab2:
            st.dataframe(sol["table"],use_container_width=True,hide_index=True)
        with tab3:
            rows=[]
            for bgt in range(60000,110001,5000):
                s,_=solve_projects(bgt,y12_budget,force_both_dc,force_p15,minp,maxp,risk_adjust)
                rows.append({"Ngân sách":bgt,"Z*":s["value"] if s else np.nan,"Số dự án":s["count"] if s else np.nan})
            sdf=pd.DataFrame(rows)
            st.plotly_chart(clean_fig(px.line(sdf,x="Ngân sách",y="Z*",markers=True,title="Độ nhạy lợi ích theo ngân sách")),use_container_width=True)
            st.dataframe(sdf,use_container_width=True,hide_index=True)
        with tab4:
            p15_chosen = "Có" if selected["Mã"].eq("P15").any() else "Không"
            policy_box("Nhận xét và hàm ý chính sách", [
                f"P15 Open Data được chọn: <b>{p15_chosen}</b>. Nếu không được chọn dù tỷ suất cao, nguyên nhân thường là ràng buộc số lượng, ngân sách năm 1–2 hoặc lợi ích tuyệt đối nhỏ so với dự án lớn.",
                "Ràng buộc P14 an ninh mạng bắt buộc là hợp lý về chính sách vì mọi hạ tầng số/AI đều làm tăng bề mặt tấn công mạng; đây là điều kiện nền chứ không chỉ là dự án sinh lời.",
                "Ràng buộc P8 ≤ P12 và P13 ≤ P12 phản ánh logic tiên quyết: trung tâm AI và bán dẫn chỉ hiệu quả khi có chương trình đào tạo kỹ sư đi kèm.",
                "Nếu Quốc hội yêu cầu cả P1 và P2 để bảo đảm dự phòng dữ liệu quốc gia, mô hình có thể vẫn khả thi nhưng sẽ hy sinh một phần lợi ích biên do giảm ngân sách cho dự án có tỷ suất cao hơn.",
                "Mở rộng nên thêm hiệu ứng cộng hưởng: ví dụ biến tương tác y8·y13 hoặc biến nhị phân z ≤ y8, z ≤ y13, z ≥ y8+y13−1 để cộng thêm lợi ích khi AI và bán dẫn cùng được chọn."
            ])

# =========================================================
# BAI 6
# =========================================================
elif page == "Bài 6 · TOPSIS vùng":
    hero("Bài 6. TOPSIS xếp hạng 6 vùng", "Xếp hạng mức độ ưu tiên đầu tư AI theo 8 tiêu chí, có lựa chọn trọng số chuyên gia hoặc entropy.", ["TOPSIS", "MCDM", "Entropy"])
    df=regions_data()
    criteria=["GRDP/người","FDI","Digital Index","AI readiness","LĐ đào tạo","R&D/GRDP","Internet","Gini"]
    with st.sidebar:
        mode=st.radio("Trọng số",["Chuyên gia","Entropy"])
        w=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
        if mode=="Chuyên gia":
            vals=[]
            for i,c in enumerate(criteria): vals.append(st.slider(c,0.0,0.4,float(w[i]),0.01))
            w=np.array(vals); w=w/w.sum()
    X=df[criteria].astype(float).values
    if mode=="Entropy":
        Xpos=X.copy(); Xpos[:,-1]=Xpos[:,-1].max()-Xpos[:,-1]+1e-9
        P=Xpos/Xpos.sum(axis=0); E=-(1/np.log(len(Xpos)))*np.nansum(P*np.log(P+1e-12),axis=0); div=1-E; w=div/div.sum()
    R=X/np.sqrt((X**2).sum(axis=0)); V=R*w
    benefit=np.array([True,True,True,True,True,True,True,False])
    Apos=np.where(benefit,V.max(axis=0),V.min(axis=0)); Aneg=np.where(benefit,V.min(axis=0),V.max(axis=0))
    Spos=np.sqrt(((V-Apos)**2).sum(axis=1)); Sneg=np.sqrt(((V-Aneg)**2).sum(axis=1)); C=Sneg/(Spos+Sneg)
    out=df[["Vùng"]].copy(); out["TOPSIS"] = C; out["Xếp hạng"] = out["TOPSIS"].rank(ascending=False).astype(int); out=out.sort_values("TOPSIS",ascending=False)
    c1,c2=st.columns(2)
    with c1: kpi("Vùng dẫn đầu", out.iloc[0]["Vùng"], f"TOPSIS {out.iloc[0]['TOPSIS']:.3f}")
    with c2: kpi("Phương pháp", mode, "trọng số đã chuẩn hóa")
    st.dataframe(out,use_container_width=True,hide_index=True)
    st.plotly_chart(clean_fig(px.bar(out.sort_values("TOPSIS"),x="TOPSIS",y="Vùng",orientation="h",text="TOPSIS",title="Xếp hạng TOPSIS")),use_container_width=True)
    policy_box("Nhận xét và hàm ý chính sách", ["TOPSIS phù hợp để chọn vùng triển khai trung tâm AI vì cân bằng nhiều tiêu chí thay vì chỉ nhìn GRDP hoặc FDI.","Nếu trọng số AI readiness cao, Đông Nam Bộ và Đồng bằng sông Hồng thường dẫn đầu; nếu nhấn mạnh bao trùm, cần cân nhắc thêm các vùng có khoảng cách số lớn.","Tương quan giữa Internet, Digital Index và AI readiness có thể làm đếm trùng lợi thế; nên kiểm tra PCA hoặc giảm trọng số nhóm tiêu chí tương quan cao."])

# =========================================================
# GENERIC INTERACTIVE PLACEHOLDERS FOR 7-12 WITH POLICY NOTES
# =========================================================
elif page == "Bài 7 · Pareto đa mục tiêu":
    hero("Bài 7. Tối ưu đa mục tiêu Pareto", "Minh họa frontier giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.", ["Pareto", "Multi-objective"])
    n=st.sidebar.slider("Số nghiệm mô phỏng",50,1000,300,50); seed=st.sidebar.number_input("Seed",0,999,42)
    rng=np.random.default_rng(seed); x=rng.dirichlet([2,2,2,2],size=n)
    growth=80*x[:,1]+60*x[:,0]+30*x[:,2]; inclusion=70*x[:,2]+45*x[:,0]-20*x[:,1]; env=65*x[:,3]-35*x[:,0]-20*x[:,1]; security=60*x[:,3]+20*x[:,2]
    df=pd.DataFrame({"Tăng trưởng":growth,"Bao trùm":inclusion,"Môi trường":env,"An ninh":security,"AI":x[:,1],"Nhân lực":x[:,2],"Hạ tầng":x[:,0],"Quản trị xanh":x[:,3]})
    score=df[["Tăng trưởng","Bao trùm","Môi trường","An ninh"]].apply(norm_good).mean(axis=1); df["Compromise score"]=score
    st.plotly_chart(clean_fig(px.scatter_3d(df,x="Tăng trưởng",y="Bao trùm",z="Môi trường",color="Compromise score",hover_data=["AI","Nhân lực","Hạ tầng","Quản trị xanh"],title="Không gian nghiệm Pareto mô phỏng"),650),use_container_width=True)
    st.dataframe(df.sort_values("Compromise score",ascending=False).head(10),use_container_width=True,hide_index=True)
    policy_box("Nhận xét và hàm ý chính sách", ["Không có một nghiệm tối ưu duy nhất; tăng trưởng nhanh có thể đánh đổi với môi trường hoặc bao trùm.","Nghiệm thỏa hiệp nên được chọn bằng quy trình minh bạch, công bố trọng số xã hội và ràng buộc không được vi phạm.","Frontier giúp nhà hoạch định chính sách hiểu chi phí cơ hội của từng mục tiêu thay vì chỉ tối đa hóa GDP."])

elif page == "Bài 8 · Tối ưu động":
    hero("Bài 8. Tối ưu liên thời gian 2026–2035", "Mô phỏng phân bổ ngân sách qua thời gian với chiết khấu phúc lợi.", ["Dynamic Programming", "Policy path"])
    years=np.arange(2026,2036); disc=st.sidebar.slider("Hệ số chiết khấu",0.85,0.99,0.95,0.01); budget=st.sidebar.slider("Ngân sách/năm",50,200,100,10)
    ai_share=st.sidebar.slider("Tỷ trọng AI ban đầu",0.1,0.6,0.3,0.05); ramp=st.sidebar.slider("Tốc độ tăng AI share",-0.02,0.05,0.01,0.005)
    shares=np.clip(ai_share+ramp*np.arange(len(years)),0.05,0.75); welfare=(budget*(1+1.8*shares)*(disc**np.arange(len(years))))
    df=pd.DataFrame({"Năm":years,"AI share":shares,"Ngân sách AI":budget*shares,"Phúc lợi chiết khấu":welfare})
    st.plotly_chart(clean_fig(px.line(df,x="Năm",y=["Ngân sách AI","Phúc lợi chiết khấu"],markers=True,title="Đường đi chính sách liên thời gian")),use_container_width=True)
    st.dataframe(df,use_container_width=True,hide_index=True)
    policy_box("Nhận xét và hàm ý chính sách", ["Đầu tư AI quá sớm khi nhân lực chưa sẵn sàng có thể kém hiệu quả; đầu tư quá muộn làm mất lợi thế tích lũy.","Chiết khấu cao ưu tiên hiện tại, chiết khấu thấp coi trọng thế hệ tương lai hơn.","Chính sách tốt cần lộ trình: hạ tầng và nhân lực trước, AI chuyên sâu tăng dần khi năng lực hấp thụ cải thiện."])

elif page == "Bài 9 · AI & lao động":
    hero("Bài 9. Tác động AI tới lao động", "Tương tác giữa tự động hóa, tạo việc mới và đào tạo lại theo ngành.", ["Labor transition", "Reskilling"])
    sectors=sectors_data().head(8).copy(); adoption=st.sidebar.slider("Tốc độ AI adoption",0.0,1.0,0.45,0.05); retrain=st.sidebar.slider("Năng lực đào tạo lại",0.0,1.0,0.35,0.05)
    sectors["Việc bị thay thế"] = sectors["Việc làm"] * sectors["Rủi ro TĐH"]/100 * adoption
    sectors["Việc mới"] = sectors["Việc làm"] * sectors["AI readiness"]/100 * adoption * 0.25
    sectors["Đào tạo hấp thụ"] = sectors["Việc bị thay thế"] * retrain
    sectors["Việc làm ròng"] = sectors["Việc mới"] + sectors["Đào tạo hấp thụ"] - sectors["Việc bị thay thế"]
    st.dataframe(sectors[["Ngành","Việc bị thay thế","Việc mới","Đào tạo hấp thụ","Việc làm ròng"]],use_container_width=True,hide_index=True)
    st.plotly_chart(clean_fig(px.bar(sectors,x="Ngành",y=["Việc bị thay thế","Việc mới","Đào tạo hấp thụ"],barmode="group",title="Tác động AI tới lao động theo ngành"),560),use_container_width=True)
    policy_box("Nhận xét và hàm ý chính sách", ["AI không chỉ thay thế lao động; tác động ròng phụ thuộc mạnh vào đào tạo lại và khả năng tạo việc bổ sung.","Ngành có rủi ro tự động hóa cao cần quỹ chuyển đổi kỹ năng trước khi mở rộng ứng dụng AI diện rộng.","Chính sách lao động nên theo ngành, không dùng một tỷ lệ hỗ trợ chung cho toàn nền kinh tế."])

elif page == "Bài 10 · Stochastic LP":
    hero("Bài 10. Quy hoạch ngẫu nhiên 2 giai đoạn", "So sánh SP, EV, WS và tính EVPI/VSS trong điều kiện bất định.", ["Stochastic Programming", "EVPI", "VSS"])
    p_high=st.sidebar.slider("Xác suất kịch bản cao",0.1,0.8,0.35,0.05); p_low=st.sidebar.slider("Xác suất kịch bản thấp",0.1,0.8,0.25,0.05); p_base=max(0,1-p_high-p_low)
    probs=np.array([p_low,p_base,p_high]); returns=np.array([0.75,1.0,1.35]); invest=np.array([40,35,25])
    sp=(probs*returns).sum()*invest.sum(); ws=(probs*np.array([max(returns)*100, max(returns)*100, max(returns)*100])).sum(); ev=returns.mean()*invest.sum();
    df=pd.DataFrame({"Chỉ tiêu":["SP","EV","WS","EVPI","VSS"],"Giá trị":[sp,ev,ws,ws-sp,sp-ev]})
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.plotly_chart(clean_fig(px.bar(df,x="Chỉ tiêu",y="Giá trị",text="Giá trị",title="So sánh các giá trị stochastic programming")),use_container_width=True)
    policy_box("Nhận xét và hàm ý chính sách", ["EVPI đo giá trị tối đa của thông tin hoàn hảo; nếu EVPI lớn, cần đầu tư dự báo và dữ liệu tốt hơn.","VSS cho biết lợi ích của mô hình ngẫu nhiên so với quyết định theo kỳ vọng đơn giản.","Với chính sách công, mô hình SP giúp tránh quyết định quá lạc quan trong kịch bản tăng trưởng AI cao."])

elif page == "Bài 11 · Q-learning":
    hero("Bài 11. Học tăng cường Q-learning", "Mô phỏng quá trình học chính sách thích nghi trong không gian trạng thái rời rạc.", ["RL", "Q-table", "Learning curve"])
    episodes=st.sidebar.slider("Episodes",200,5000,1500,100); alpha=st.sidebar.slider("α learning rate",0.01,0.5,0.10,0.01); gamma=st.sidebar.slider("γ discount",0.5,0.99,0.95,0.01); seed=st.sidebar.number_input("Seed",0,999,42)
    rng=np.random.default_rng(seed); trend=-20+8*(1-np.exp(-np.arange(episodes)/max(50,episodes/5))); rewards=trend+rng.normal(0,2,episodes); smooth=pd.Series(rewards).rolling(50,min_periods=1).mean()
    df=pd.DataFrame({"Episode":np.arange(episodes),"Reward":rewards,"Smoothed":smooth})
    c1,c2=st.columns(2)
    with c1: kpi("Mean reward 100 ep cuối", f"{rewards[-100:].mean():.2f}", "càng cao càng tốt")
    with c2: kpi("Tham số", f"α={alpha:.2f}, γ={gamma:.2f}", "tabular Q-learning")
    st.plotly_chart(clean_fig(px.line(df,x="Episode",y=["Reward","Smoothed"],title="Learning curve mô phỏng"),520),use_container_width=True)
    policy_box("Nhận xét và hàm ý chính sách", ["Q-learning phù hợp khi chính sách cần học dần từ phản hồi, ví dụ điều chỉnh mức đầu tư AI theo trạng thái rủi ro thất nghiệp và GDP.","Reward phải phản ánh đa mục tiêu: tăng GDP, giảm thất nghiệp, giảm rủi ro mạng và kiểm soát phát thải.","Không nên dùng RL như hộp đen; cần ràng buộc an toàn và giải thích chính sách trước khi áp dụng vào quản trị công."])

elif page == "Bài 12 · Tích hợp":
    hero("Bài 12. Dashboard tích hợp AIDEOM-VN", "Tổng hợp 6 module và so sánh 5 kịch bản chính sách 2026–2030.", ["Integrated dashboard", "Scenario comparison"])
    scenario=st.sidebar.selectbox("Kịch bản",["Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm","Xanh & an toàn"])
    base=pd.DataFrame({
        "Kịch bản":["Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm","Xanh & an toàn"],
        "GDP gain":[100,115,128,92,88],"Việc làm ròng":[20,10,5,35,18],"Rủi ro mạng":[35,48,62,30,22],"Phát thải":[45,50,55,38,25],"Điểm tổng hợp":[82,78,75,84,86]
    })
    st.dataframe(base,use_container_width=True,hide_index=True)
    st.plotly_chart(clean_fig(px.bar(base,x="Kịch bản",y=["GDP gain","Việc làm ròng","Điểm tổng hợp"],barmode="group",title="So sánh 5 kịch bản chính sách")),use_container_width=True)
    row=base[base["Kịch bản"]==scenario].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi("GDP gain", row["GDP gain"], scenario)
    with c2: kpi("Việc làm ròng", row["Việc làm ròng"], scenario)
    with c3: kpi("Rủi ro mạng", row["Rủi ro mạng"], "càng thấp càng tốt")
    with c4: kpi("Điểm tổng hợp", row["Điểm tổng hợp"], "dashboard score")
    policy_box("Nhận xét và hàm ý chính sách", ["Kịch bản AI dẫn dắt tạo GDP gain cao nhưng kéo theo rủi ro mạng và rủi ro lao động lớn hơn.","Kịch bản bao trùm cải thiện việc làm ròng nhưng có thể hy sinh một phần tăng trưởng ngắn hạn.","Kịch bản xanh & an toàn có điểm tổng hợp cao khi xã hội coi trọng bền vững, an ninh dữ liệu và ổn định dài hạn.","AIDEOM-VN nên được dùng như hệ hỗ trợ ra quyết định: minh bạch giả định, cho phép thay trọng số, và trình bày rõ đánh đổi chính sách."])
