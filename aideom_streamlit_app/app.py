
# app.py — Dashboard Streamlit cho AIDEOM-VN
# Chạy local: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

st.set_page_config(
    page_title="AIDEOM-VN | Mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.main .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid #e7eaf0; border-radius: 18px; padding: 18px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    box-shadow: 0 6px 20px rgba(15,23,42,0.06);
}
.small-note {color: #64748b; font-size: 0.9rem;}
.big-title {font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;}
.section-title {font-size: 1.35rem; font-weight: 750; margin-top: .6rem;}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data
def load_macro():
    return pd.DataFrame({
        "year":[2020,2021,2022,2023,2024,2025],
        "GDP_trillion_VND":[8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
        "GDP_growth_pct":[2.91,2.58,8.02,5.05,7.09,8.02],
        "GDP_per_capita_USD":[3521,3717,4163,4347,4700,5026],
        "population_million":[97.58,98.51,99.46,100.30,101.30,102.30],
        "digital_economy_share_GDP_pct":[12.0,12.7,14.3,16.5,18.3,19.5],
        "FDI_disbursed_billion_USD":[19.98,19.74,22.40,23.18,25.35,27.60],
        "exports_billion_USD":[282.6,336.3,371.3,355.5,405.5,475.0],
        "labor_productivity_million_VND":[151.2,171.3,188.1,199.3,221.9,245.0],
    })

@st.cache_data
def load_sectors():
    return pd.DataFrame({
        "sector_name_vi":[
            "Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Khai khoáng",
            "Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải",
            "CNTT-Truyền thông","Giáo dục-Đào tạo","Y tế"
        ],
        "growth_rate_2024_pct":[3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
        "productivity_million_VND_per_worker":[103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1],
        "spillover_coef_0_1":[0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
        "export_billion_USD":[40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
        "labor_million":[13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
        "ai_readiness_0_100":[15,55,20,30,48,72,42,88,38,45],
        "automation_risk_pct":[18,42,25,55,38,52,35,28,22,18],
    })

@st.cache_data
def load_regions():
    return pd.DataFrame({
        "region_code":["NMM","RRD","NCC","CH","SE","MD"],
        "region_name_vi":[
            "Trung du miền núi phía Bắc","Đồng bằng sông Hồng",
            "Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên",
            "Đông Nam Bộ","Đồng bằng sông Cửu Long"
        ],
        "grdp_per_capita_million_VND":[57.0,152.3,87.5,68.9,158.9,80.5],
        "fdi_registered_billion_USD":[3.5,20.0,8.2,0.8,18.5,2.1],
        "digital_index_0_100":[38,78,55,32,82,48],
        "ai_readiness_0_100":[22,68,40,18,75,30],
        "trained_labor_pct":[21.5,36.8,27.5,18.2,42.5,16.8],
        "rd_intensity_pct":[0.18,0.85,0.32,0.15,0.78,0.22],
        "internet_penetration_pct":[72,92,84,68,94,78],
        "gini_coef":[0.405,0.358,0.372,0.412,0.385,0.392],
    })

def norm_good(x):
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else x*0

def norm_bad(x):
    rng = x.max() - x.min()
    return (x.max() - x) / rng if rng else x*0

def plot_table(df, height=420):
    st.dataframe(df, use_container_width=True, height=height)

def policy_box(title, text):
    st.info(f"**{title}**\n\n{text}")

macro = load_macro()
sectors = load_sectors()
regions = load_regions()

# ---------- SIDEBAR ----------
st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption("Dashboard Streamlit cho bài cuối kỳ Mô hình ra quyết định")
page = st.sidebar.radio(
    "Chọn mô-đun",
    ["Tổng quan","Bài 1 Cobb-Douglas","Bài 2 LP ngân sách","Bài 3 Priority ngành",
     "Bài 4 LP vùng × hạng mục","Bài 5 MIP dự án","Bài 6 TOPSIS vùng",
     "Bài 7 Pareto","Bài 8 Tối ưu động","Bài 9 Lao động & AI",
     "Bài 10 Ngẫu nhiên","Bài 11 Q-learning","Bài 12 Kịch bản tổng hợp"],
)

st.sidebar.markdown("---")
st.sidebar.write("**Gợi ý khi nộp:** ghi rõ đây là dashboard minh họa, kèm file `.ipynb`/`.py` gốc và diễn giải chính sách.")

# ---------- PAGES ----------
if page == "Tổng quan":
    st.markdown('<div class="big-title">Phát triển kinh tế Việt Nam trong kỉ nguyên AI</div>', unsafe_allow_html=True)
    st.caption("Mô hình AIDEOM-VN · Python · Tối ưu hóa · Học tăng cường · Streamlit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", f"{macro.GDP_trillion_VND.iloc[-1]:,.1f}", "nghìn tỷ VND")
    c2.metric("Tăng trưởng GDP 2025", f"{macro.GDP_growth_pct.iloc[-1]:.2f}%")
    c3.metric("Kinh tế số/GDP", f"{macro.digital_economy_share_GDP_pct.iloc[-1]:.1f}%")
    c4.metric("FDI giải ngân", f"{macro.FDI_disbursed_billion_USD.iloc[-1]:.1f} tỷ USD")
    col1, col2 = st.columns([1.15, .85])
    with col1:
        fig = px.line(macro, x="year", y=["GDP_trillion_VND","labor_productivity_million_VND"],
                      markers=True, title="Xu hướng GDP và năng suất lao động 2020–2025")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(macro, x="year", y="digital_economy_share_GDP_pct",
                     title="Tỷ trọng kinh tế số/GDP (%)")
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Xem dữ liệu vĩ mô"):
        plot_table(macro)

elif page == "Bài 1 Cobb-Douglas":
    st.header("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng")
    alpha = st.sidebar.slider("α — vốn K", 0.05, 0.60, 0.33, 0.01)
    beta = st.sidebar.slider("β — lao động L", 0.05, 0.70, 0.42, 0.01)
    gamma = st.sidebar.slider("γ — số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = st.sidebar.slider("δ — AI", 0.01, 0.25, 0.08, 0.01)
    theta = max(0.0, 1 - alpha - beta - gamma - delta)
    st.sidebar.metric("θ tự động để tổng = 1", f"{theta:.2f}")

    Y = macro["GDP_trillion_VND"].values
    K = np.array([16500,17800,19600,21300,23500,25900], dtype=float)
    L = np.array([53.6,50.5,51.7,52.4,52.9,53.4], dtype=float)
    D = np.array([12.0,12.7,14.3,16.5,18.3,19.5], dtype=float)
    AI = np.array([55.6,60.2,65.4,67.0,73.8,80.1], dtype=float)
    H = np.array([24.1,26.1,26.2,27.0,28.4,29.2], dtype=float)

    A = Y/(K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    A_mean = A.mean()
    Y_hat = A_mean*(K**alpha * L**beta * D**gamma * AI**delta * H**theta)
    mape = np.mean(np.abs((Y-Y_hat)/Y))*100
    out = pd.DataFrame({"Năm":macro.year, "GDP thực tế":Y, "TFP A_t":A, "GDP dự báo":Y_hat, "Sai số %":(Y_hat/Y-1)*100})
    c1,c2,c3 = st.columns(3)
    c1.metric("TFP 2020", f"{A[0]:.2f}")
    c2.metric("TFP 2025", f"{A[-1]:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%")
    col1,col2 = st.columns(2)
    with col1:
        fig = px.line(out, x="Năm", y="TFP A_t", markers=True, title="Xu hướng TFP A_t")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(out, x="Năm", y=["GDP thực tế","GDP dự báo"], markers=True, title="GDP thực tế vs dự báo")
        st.plotly_chart(fig, use_container_width=True)

    gY = np.log(Y[-1]/Y[0])/5
    parts = {
        "Vốn K": alpha*np.log(K[-1]/K[0])/5,
        "Lao động L": beta*np.log(L[-1]/L[0])/5,
        "Số hóa D": gamma*np.log(D[-1]/D[0])/5,
        "AI": delta*np.log(AI[-1]/AI[0])/5,
        "Nhân lực H": theta*np.log(H[-1]/H[0])/5,
    }
    parts["TFP"] = gY - sum(parts.values())
    dc = pd.DataFrame({"Yếu tố":list(parts.keys()), "Điểm %/năm":[v*100 for v in parts.values()]})
    fig = px.bar(dc, x="Điểm %/năm", y="Yếu tố", orientation="h", title="Phân rã tăng trưởng bình quân")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Bảng kết quả"):
        plot_table(out.round(3))
    policy_box("Hàm ý", "TFP và số hóa là hai phần cần giải thích kỹ trong báo cáo: nếu TFP tăng, chất lượng tăng trưởng cải thiện; nếu D tăng nhanh, mục tiêu kinh tế số có vai trò đáng kể trong tăng trưởng.")

elif page == "Bài 2 LP ngân sách":
    st.header("Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách")
    B = st.sidebar.slider("Ngân sách tổng (nghìn tỷ VND)", 80, 180, 100, 5)
    x3_min = st.sidebar.slider("Sàn nhân lực số x3", 10, 50, 20, 1)
    c = [-0.85,-1.20,-0.95,-1.35]
    A_ub = [
        [1,1,1,1],
        [-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],
        [0.35,-0.65,0.35,-0.65]
    ]
    b_ub = [B,-25,-15,-x3_min,-10,0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*4, method="highs")
    if res.success:
        names = ["Hạ tầng số","AI & dữ liệu","Nhân lực số","R&D công nghệ"]
        df = pd.DataFrame({"Hạng mục":names, "Phân bổ":res.x, "Hệ số tác động":[0.85,1.20,0.95,1.35]})
        st.metric("GDP gain tối ưu Z*", f"{-res.fun:.2f} nghìn tỷ VND")
        st.plotly_chart(px.bar(df, x="Hạng mục", y="Phân bổ", text_auto=".1f", title="Phân bổ tối ưu"), use_container_width=True)
        Bs = np.arange(80,181,10)
        Zs = []
        for b in Bs:
            rr = linprog(c, A_ub=A_ub, b_ub=[b,-25,-15,-x3_min,-10,0], bounds=[(0,None)]*4, method="highs")
            Zs.append(-rr.fun if rr.success else np.nan)
        st.plotly_chart(px.line(pd.DataFrame({"B":Bs,"Z*":Zs}), x="B", y="Z*", markers=True, title="Đường cong Z*(B)"), use_container_width=True)
        plot_table(df.round(3), 220)
    else:
        st.error("Bài toán không khả thi với tham số hiện tại.")

elif page == "Bài 3 Priority ngành":
    st.header("Bài 3 — Chỉ số ưu tiên ngành")
    cols = ["growth_rate_2024_pct","productivity_million_VND_per_worker","spillover_coef_0_1",
            "export_billion_USD","labor_million","ai_readiness_0_100"]
    labels = ["Tăng trưởng","Năng suất","Lan tỏa","Xuất khẩu","Việc làm","AI readiness"]
    weights = []
    st.sidebar.write("Trọng số các tiêu chí tốt")
    defaults = [0.15,0.15,0.20,0.15,0.10,0.20]
    for lab, d in zip(labels, defaults):
        weights.append(st.sidebar.slider(lab, 0.0, 0.5, d, 0.01))
    w_risk = st.sidebar.slider("Phạt rủi ro tự động hóa", 0.0, 0.5, 0.15, 0.01)
    Xg = sectors[cols].apply(norm_good)
    Xb = norm_bad(sectors["automation_risk_pct"])
    w = np.array(weights)
    if w.sum() > 0:
        # giữ đúng tinh thần trọng số, không ép tổng 1 để người học thấy tác động
        pr = Xg.values @ w - w_risk*Xb.values
    else:
        pr = np.zeros(len(sectors))
    out = sectors[["sector_name_vi"]].copy()
    out["Priority"] = pr
    out = out.sort_values("Priority", ascending=False)
    st.plotly_chart(px.bar(out, x="Priority", y="sector_name_vi", orientation="h",
                           title="Xếp hạng ưu tiên chuyển đổi số theo ngành"), use_container_width=True)
    st.subheader("Top 3")
    st.write(", ".join(out.head(3)["sector_name_vi"].tolist()))
    plot_table(out.round(4), 360)

elif page == "Bài 4 LP vùng × hạng mục":
    st.header("Bài 4 — Phân bổ ngân sách số theo vùng × hạng mục")
    budget = st.sidebar.slider("Ngân sách tổng (tỷ VND)", 30000, 70000, 50000, 1000)
    floor = st.sidebar.slider("Sàn mỗi vùng", 2000, 8000, 5000, 500)
    cap = st.sidebar.slider("Trần mỗi vùng", 8000, 16000, 12000, 500)
    fairness = st.sidebar.slider("λ công bằng", 0.50, 0.95, 0.70, 0.05)

    regs = regions.region_code.tolist()
    items = ["I","D","AI","H"]
    beta = np.array([
        [1.15,0.85,0.55,1.30],
        [0.95,1.25,1.40,1.05],
        [1.05,0.95,0.85,1.15],
        [1.20,0.75,0.45,1.35],
        [0.90,1.30,1.55,1.00],
        [1.10,0.85,0.65,1.25],
    ])
    # LP với biến x(24) và M(1), maximize beta*x -> minimize -beta*x
    n = 24
    c = np.r_[-beta.flatten(), 0.0]
    A=[]; b=[]
    A.append(np.r_[np.ones(n),0]); b.append(budget)
    for r in range(6):
        row = np.zeros(n+1); row[r*4:(r+1)*4] = -1; A.append(row); b.append(-floor)
        row = np.zeros(n+1); row[r*4:(r+1)*4] = 1; A.append(row); b.append(cap)
    row = np.zeros(n+1); row[3::4] = -1; A.append(row); b.append(-12000)
    D0 = regions.digital_index_0_100.values
    gamma = 0.002
    for r in range(6):
        row = np.zeros(n+1); row[r*4+1] = gamma; row[-1] = -1
        A.append(row); b.append(-D0[r])  # D0 + gamma*xD <= M -> gamma*xD - M <= -D0
    for r in range(6):
        row = np.zeros(n+1); row[r*4+1] = -gamma; row[-1] = fairness
        A.append(row); b.append(D0[r])   # D0+gamma*xD >= lam*M -> -gamma*xD+lam*M <= D0
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0,None)]*(n+1), method="highs")
    if res.success:
        X = res.x[:n].reshape(6,4)
        df = pd.DataFrame(X, columns=["Hạ tầng I","CĐS D","AI","Nhân lực H"], index=regions.region_name_vi)
        st.metric("GDP gain tối ưu Z*", f"{-res.fun:.2f}")
        st.plotly_chart(px.imshow(df, text_auto=".0f", aspect="auto", title="Heatmap phân bổ tối ưu"), use_container_width=True)
        plot_table(df.round(1), 300)
    else:
        st.error("Không khả thi. Hãy giảm sàn vùng, tăng ngân sách hoặc giảm λ.")

elif page == "Bài 5 MIP dự án":
    st.header("Bài 5 — Quy hoạch nguyên hỗn hợp chọn dự án")
    budget = st.sidebar.slider("Ngân sách 5 năm", 60000, 110000, 80000, 5000)
    early = st.sidebar.slider("Ngân sách năm 1–2", 30000, 60000, 40000, 2500)
    P = list(range(1,16))
    C = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    names = {
        1:"Data Center Hòa Lạc",2:"Data Center phía Nam",3:"5G toàn quốc",4:"VNeID 2.0",5:"DVC quốc gia v3",
        6:"Y tế số",7:"Giáo dục số K-12",8:"Trung tâm AI quốc gia",9:"Sandbox fintech",10:"Logistics thông minh",
        11:"Nông nghiệp số ĐBSCL",12:"Đào tạo kỹ sư AI/bán dẫn",13:"Khu CN bán dẫn",14:"An ninh mạng SOC",15:"Open Data"
    }
    if not HAS_PULP:
        st.warning("Chưa cài PuLP. Hãy thêm pulp vào requirements.txt.")
    else:
        m = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", P, cat="Binary")
        m += pulp.lpSum(B[i]*y[i] for i in P)
        m += pulp.lpSum(C[i]*y[i] for i in P) <= budget
        m += pulp.lpSum(C1[i]*y[i] for i in P) <= early
        m += y[1] + y[2] <= 1
        m += y[8] <= y[12]
        m += y[13] <= y[12]
        m += y[4] + y[5] >= 1
        m += y[14] >= 1
        m += pulp.lpSum(y[i] for i in P) >= 7
        m += pulp.lpSum(y[i] for i in P) <= 11
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [i for i in P if y[i].value() and y[i].value() > 0.5]
        df = pd.DataFrame({"Mã":[f"P{i}" for i in selected], "Dự án":[names[i] for i in selected],
                           "Chi phí":[C[i] for i in selected], "NPV":[B[i] for i in selected]})
        st.metric("Tổng NPV", f"{sum(B[i] for i in selected):,.0f} tỷ VND")
        st.metric("Tổng chi phí", f"{sum(C[i] for i in selected):,.0f} tỷ VND")
        st.plotly_chart(px.bar(df, x="Mã", y="NPV", hover_name="Dự án", title="Dự án được chọn"), use_container_width=True)
        plot_table(df, 360)

elif page == "Bài 6 TOPSIS vùng":
    st.header("Bài 6 — TOPSIS xếp hạng vùng ưu tiên AI")
    criteria = ["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100","ai_readiness_0_100",
                "trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
    labels = ["GRDP/người","FDI","Digital","AI","LĐ đào tạo","R&D","Internet","Gini"]
    w = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    ai_weight = st.sidebar.slider("Trọng số AI readiness", 0.05, 0.45, 0.20, 0.05)
    w[3] = ai_weight
    w = w/w.sum()
    is_benefit = np.array([True,True,True,True,True,True,True,False])
    X = regions[criteria].values.astype(float)
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R*w
    Astar = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    Aneg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    Sstar = np.sqrt(((V-Astar)**2).sum(axis=1))
    Sneg = np.sqrt(((V-Aneg)**2).sum(axis=1))
    Cstar = Sneg/(Sstar+Sneg)
    out = regions[["region_name_vi"]].copy()
    out["TOPSIS_score"] = Cstar
    out = out.sort_values("TOPSIS_score", ascending=False)
    st.plotly_chart(px.bar(out, x="TOPSIS_score", y="region_name_vi", orientation="h",
                           title="TOPSIS score theo vùng"), use_container_width=True)
    plot_table(out.round(4), 260)

elif page == "Bài 7 Pareto":
    st.header("Bài 7 — Minh họa tập nghiệm Pareto đa mục tiêu")
    n = st.sidebar.slider("Số phương án mô phỏng", 100, 2000, 600, 100)
    rng = np.random.default_rng(42)
    X = rng.dirichlet([1.2,1.1,1.0,1.0], size=n)
    growth = 0.9*X[:,0] + 1.3*X[:,1] + 1.0*X[:,2] + 1.4*X[:,3]
    inclusion = 1.5*X[:,2] + 0.9*X[:,0] - 0.4*X[:,1]
    env = 1.2*X[:,0] + 1.1*X[:,2] - 0.5*X[:,3]
    security = 0.8*X[:,0] + 1.5*X[:,1] + 0.6*X[:,3]
    df = pd.DataFrame({"Growth":growth,"Bao trùm":inclusion,"Môi trường":env,"An ninh dữ liệu":security,
                       "I":X[:,0],"AI":X[:,1],"H":X[:,2],"R&D":X[:,3]})
    score = df[["Growth","Bao trùm","Môi trường","An ninh dữ liệu"]].rank(pct=True).mean(axis=1)
    df["Compromise score"] = score
    st.plotly_chart(px.scatter_3d(df, x="Growth", y="Bao trùm", z="Môi trường",
                                  color="Compromise score", hover_data=["I","AI","H","R&D"],
                                  title="Không gian phương án Pareto/thoả hiệp"), use_container_width=True)
    plot_table(df.nlargest(10, "Compromise score").round(3), 300)
    policy_box("Cách dùng trong báo cáo", "Không nên chỉ chọn phương án tăng trưởng cao nhất. Hãy chọn nghiệm thỏa hiệp cân bằng tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.")

elif page == "Bài 8 Tối ưu động":
    st.header("Bài 8 — Quỹ đạo đầu tư 2026–2035")
    inv = st.sidebar.slider("Tỷ lệ đầu tư số mỗi năm (% GDP)", 1.0, 8.0, 4.0, 0.5)/100
    tfp = st.sidebar.slider("TFP tăng/năm", 0.0, 3.0, 1.2, 0.1)/100
    years = np.arange(2025,2036)
    gdp = [macro.GDP_trillion_VND.iloc[-1]]
    digital = [macro.digital_economy_share_GDP_pct.iloc[-1]]
    for t in years[1:]:
        digital.append(min(45, digital[-1] + 1.4 + inv*45))
        g = 0.045 + tfp + 0.002*(digital[-1]-digital[-2])
        gdp.append(gdp[-1]*(1+g))
    df = pd.DataFrame({"Năm":years,"GDP":gdp,"Kinh tế số/GDP (%)":digital})
    st.plotly_chart(px.line(df, x="Năm", y=["GDP","Kinh tế số/GDP (%)"], markers=True,
                            title="Quỹ đạo mô phỏng đến 2035"), use_container_width=True)
    plot_table(df.round(2), 360)

elif page == "Bài 9 Lao động & AI":
    st.header("Bài 9 — Tác động AI tới lao động theo ngành")
    reskill = st.sidebar.slider("Tỷ lệ tái đào tạo thành công", 0.0, 1.0, 0.45, 0.05)
    df = sectors.copy()
    df["LĐ có rủi ro (triệu)"] = df.labor_million * df.automation_risk_pct/100
    df["LĐ chuyển đổi thành công"] = df["LĐ có rủi ro (triệu)"] * reskill
    df["LĐ còn rủi ro"] = df["LĐ có rủi ro (triệu)"] - df["LĐ chuyển đổi thành công"]
    st.plotly_chart(px.bar(df.sort_values("LĐ còn rủi ro", ascending=False),
                           x="sector_name_vi", y=["LĐ chuyển đổi thành công","LĐ còn rủi ro"],
                           title="Mô phỏng rủi ro việc làm và tái đào tạo", barmode="stack"), use_container_width=True)
    plot_table(df[["sector_name_vi","labor_million","automation_risk_pct","LĐ còn rủi ro"]].round(3), 360)

elif page == "Bài 10 Ngẫu nhiên":
    st.header("Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn")
    st.write("Mô phỏng 3 kịch bản: thuận lợi, cơ sở, bất lợi.")
    probs = np.array([0.25,0.50,0.25])
    scen = pd.DataFrame({"Kịch bản":["Thuận lợi","Cơ sở","Bất lợi"], "Xác suất":probs, "Hệ số hiệu quả":[1.20,1.00,0.78]})
    x = np.array([25,20,25,30])
    coef = np.array([0.85,1.20,0.95,1.35])
    scen["GDP gain"] = [p for p in scen["Hệ số hiệu quả"].values*(x@coef)]
    st.metric("Kỳ vọng GDP gain", f"{np.dot(scen['Xác suất'], scen['GDP gain']):.2f}")
    st.plotly_chart(px.bar(scen, x="Kịch bản", y="GDP gain", text_auto=".1f", title="GDP gain theo kịch bản"), use_container_width=True)
    plot_table(scen.round(3), 220)

elif page == "Bài 11 Q-learning":
    st.header("Bài 11 — Tabular Q-learning minh họa")
    episodes = st.sidebar.slider("Số episode", 100, 3000, 1000, 100)
    rng = np.random.default_rng(1)
    states, actions = 5, 3
    Q = np.zeros((states,actions))
    rewards = []
    for ep in range(episodes):
        s = rng.integers(states)
        total = 0
        eps = max(0.05, 0.6*(1-ep/episodes))
        for _ in range(12):
            a = rng.integers(actions) if rng.random()<eps else int(np.argmax(Q[s]))
            ns = min(states-1, max(0, s + (a-1) + rng.choice([-1,0,1], p=[.15,.7,.15])))
            r = (ns - s)*0.3 + (0.2 if a==1 else 0) - 0.03*a
            Q[s,a] += 0.15*(r + 0.95*np.max(Q[ns]) - Q[s,a])
            s = ns; total += r
        rewards.append(total)
    df = pd.DataFrame({"Episode":np.arange(episodes), "Reward rolling":pd.Series(rewards).rolling(50).mean()})
    st.plotly_chart(px.line(df, x="Episode", y="Reward rolling", title="Q-learning: reward trung bình trượt"), use_container_width=True)
    st.write("Bảng Q cuối cùng")
    plot_table(pd.DataFrame(Q, columns=["Giảm đầu tư","Giữ ổn định","Tăng đầu tư"]).round(3), 240)

elif page == "Bài 12 Kịch bản tổng hợp":
    st.header("Bài 12 — Dashboard kịch bản chính sách AIDEOM-VN")
    scenarios = pd.DataFrame({
        "Kịch bản":["Cơ sở","Tăng trưởng số","Bao trùm","AI mạnh","Cân bằng bền vững"],
        "GDP 2030":[16500,18400,17200,19000,18100],
        "Kinh tế số/GDP":[27,35,30,38,34],
        "Việc làm chất lượng":[62,66,75,68,73],
        "Công bằng vùng":[58,55,76,60,72],
        "Rủi ro dữ liệu":[42,48,38,55,40],
    })
    chosen = st.multiselect("Chọn kịch bản để so sánh", scenarios["Kịch bản"].tolist(),
                            default=["Cơ sở","AI mạnh","Cân bằng bền vững"])
    df = scenarios[scenarios["Kịch bản"].isin(chosen)]
    st.plotly_chart(px.bar(df, x="Kịch bản", y="GDP 2030", text_auto=True, title="GDP 2030 theo kịch bản"), use_container_width=True)
    radar_cols = ["Kinh tế số/GDP","Việc làm chất lượng","Công bằng vùng"]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(r=[row[c] for c in radar_cols]+[row[radar_cols[0]]],
                                      theta=radar_cols+[radar_cols[0]], fill="toself", name=row["Kịch bản"]))
    fig.update_layout(title="Radar so sánh mục tiêu chính sách", polar=dict(radialaxis=dict(visible=True, range=[0,100])))
    st.plotly_chart(fig, use_container_width=True)
    plot_table(scenarios, 260)
    policy_box("Kết luận gợi ý", "Kịch bản AI mạnh cho GDP cao nhưng rủi ro dữ liệu lớn; kịch bản cân bằng bền vững thường dễ bảo vệ hơn khi viết phần hàm ý chính sách.")
