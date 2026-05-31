from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.optimize import linprog, minimize
except Exception:  # pragma: no cover
    linprog = None
    minimize = None

try:
    import pulp
except Exception:  # pragma: no cover
    pulp = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="AIDEOM-VN Streamlit",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 4rem;}
    h1, h2, h3 {color: #1a2849;}
    .stMetric {background: #fffaf0; border: 1px solid rgba(26,40,73,.08); border-radius: 14px; padding: 14px;}
    div[data-testid="stSidebar"] {background: #1a2849;}
    div[data-testid="stSidebar"] * {color: #f5f1e8;}
    .small-note {font-size: 0.9rem; color: #6b6b6b;}
    .card {border: 1px solid rgba(26,40,73,.12); border-radius: 14px; padding: 16px; background: #fffaf0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_macro() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv")


@st.cache_data
def load_sectors() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv")


@st.cache_data
def load_regions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv")


def metric_row(items: List[Tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def minmax_good(s: pd.Series) -> pd.Series:
    den = s.max() - s.min()
    return (s - s.min()) / den if den else s * 0


def minmax_bad(s: pd.Series) -> pd.Series:
    den = s.max() - s.min()
    return (s.max() - s) / den if den else s * 0


def topsis(X: np.ndarray, weights: np.ndarray, is_benefit: List[bool]) -> np.ndarray:
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1
    R = X / denom
    V = R * weights
    is_benefit_arr = np.array(is_benefit, dtype=bool)
    a_star = np.where(is_benefit_arr, V.max(axis=0), V.min(axis=0))
    a_neg = np.where(is_benefit_arr, V.min(axis=0), V.max(axis=0))
    s_star = np.sqrt(((V - a_star) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - a_neg) ** 2).sum(axis=1))
    return s_neg / (s_star + s_neg + 1e-12)


def entropy_weights(X: np.ndarray) -> np.ndarray:
    X = np.maximum(X.astype(float), 1e-12)
    P = X / X.sum(axis=0, keepdims=True)
    k = 1 / np.log(len(X))
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def render_header() -> None:
    st.sidebar.markdown("# AIDEOM-VN")
    st.sidebar.caption("Phát triển kinh tế Việt Nam trong kỉ nguyên AI")
    st.sidebar.divider()


PAGES = [
    "Bài 1 · Cobb-Douglas",
    "Bài 2 · LP ngân sách 4 hạng mục",
    "Bài 3 · Priority Index 10 ngành",
    "Bài 4 · LP ngành-vùng",
    "Bài 5 · MIP chọn dự án",
    "Bài 6 · TOPSIS 6 vùng",
    "Bài 7 · Pareto đa mục tiêu",
    "Bài 8 · Tối ưu động 2026-2035",
    "Bài 9 · AI và lao động",
    "Bài 10 · Quy hoạch ngẫu nhiên",
    "Bài 11 · Q-learning",
    "Bài 12 · Dashboard tổng hợp",
]


def page1() -> Dict[str, float]:
    st.title("Bài 1. Hàm sản xuất Cobb-Douglas mở rộng")
    st.caption("M1 · Dự báo kinh tế vĩ mô 2020-2025 và mô phỏng 2030")
    df = load_macro().copy()

    st.subheader("Dữ liệu đầu vào")
    st.dataframe(df, use_container_width=True)

    alpha = st.slider("α · Vốn vật chất K", 0.10, 0.60, 0.33, 0.01)
    beta = st.slider("β · Lao động L", 0.10, 0.70, 0.42, 0.01)
    gamma = st.slider("γ · Số hóa D", 0.00, 0.30, 0.10, 0.01)
    delta = st.slider("δ · AI", 0.00, 0.30, 0.08, 0.01)
    theta = max(0.0, 1 - alpha - beta - gamma - delta)
    st.info(f"θ · Nhân lực số H được tự cân bằng theo lợi suất không đổi: **{theta:.2f}**")

    # Nếu CSV thiếu các biến K, L, AI, H thì dùng chuỗi trong đề.
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], dtype=float)
    D = df["digital_economy_share_GDP_pct"].values.astype(float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], dtype=float)
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], dtype=float)
    Y = df["GDP_trillion_VND"].values.astype(float)

    A = Y / (K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta)
    A_avg = A.mean()
    y_hat = A_avg * K ** alpha * L ** beta * D ** gamma * AI ** delta * H ** theta
    mape = np.mean(np.abs((Y - y_hat) / Y)) * 100

    metric_row([
        ("TFP trung bình", f"{A_avg:.3f}", None),
        ("MAPE dự báo", f"{mape:.2f}%", None),
        ("GDP 2025 thực tế", f"{Y[-1]:,.1f}", "nghìn tỷ VND"),
    ])

    tfp_df = pd.DataFrame({"year": df["year"], "TFP A_t": A, "GDP thực tế": Y, "GDP dự báo": y_hat})
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(tfp_df, x="year", y="TFP A_t", markers=True, title="TFP theo năm"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(tfp_df, x="year", y=["GDP thực tế", "GDP dự báo"], markers=True, title="So sánh GDP"), use_container_width=True)

    st.subheader("Kịch bản 2030")
    c1, c2, c3, c4 = st.columns(4)
    D2030 = c1.slider("D 2030 (% GDP)", 20.0, 40.0, 30.0, 0.5)
    AI2030 = c2.slider("AI 2030 (nghìn DN)", 80.0, 150.0, 100.0, 1.0)
    H2030 = c3.slider("H 2030 (% LĐ đào tạo)", 25.0, 45.0, 35.0, 0.5)
    tfp_g = c4.slider("TFP tăng/năm", 0.0, 0.04, 0.012, 0.001)
    K2030 = K[-1] * (1.06 ** 5)
    L2030 = L[-1] * (1.006 ** 5)
    A2030 = A[-1] * ((1 + tfp_g) ** 5)
    Y2030 = A2030 * K2030 ** alpha * L2030 ** beta * D2030 ** gamma * AI2030 ** delta * H2030 ** theta
    st.metric("GDP dự báo 2030", f"{Y2030:,.1f} nghìn tỷ VND", f"+{(Y2030 / Y[-1] - 1) * 100:.1f}% so với 2025")
    return {"gdp2030": float(Y2030), "mape": float(mape)}


def page2() -> Dict[str, float]:
    st.title("Bài 2. Quy hoạch tuyến tính phân bổ ngân sách")
    st.caption("LP 4 hạng mục: hạ tầng số, AI-dữ liệu, nhân lực số, R&D")

    budget = st.slider("Tổng ngân sách B (nghìn tỷ VND)", 80, 160, 100, 10)
    min_h = st.slider("Sàn nhân lực số x₃", 20, 50, 20, 5)
    coeff = np.array([0.85, 1.20, 0.95, 1.35])
    c = -coeff
    A_ub = np.array([
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ], dtype=float)
    b_ub = np.array([budget, -25, -15, -min_h, -10, 0], dtype=float)

    if linprog is None:
        st.error("Thiếu scipy. Hãy cài `scipy` trong requirements.txt.")
        return {}
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    if not res.success:
        st.error("Bài toán không khả thi với tham số hiện tại.")
        return {}
    x = res.x
    z = coeff @ x
    out = pd.DataFrame({"Hạng mục": ["I - hạ tầng số", "AI - dữ liệu", "H - nhân lực", "R&D"], "Phân bổ": x, "Hệ số": coeff})
    metric_row([("Z*", f"{z:,.2f}", "nghìn tỷ GDP kỳ vọng"), ("Ngân sách dùng", f"{x.sum():,.1f}", "nghìn tỷ"), ("Công nghệ chiến lược", f"{(x[1]+x[3])/x.sum()*100:.1f}%", "AI + R&D")])
    st.dataframe(out, use_container_width=True)
    st.plotly_chart(px.bar(out, x="Hạng mục", y="Phân bổ", title="Phân bổ tối ưu"), use_container_width=True)

    Bs = np.arange(100, 151, 10)
    zs = []
    for B in Bs:
        b_ub2 = b_ub.copy(); b_ub2[0] = B
        r = linprog(c, A_ub=A_ub, b_ub=b_ub2, bounds=[(0, None)] * 4, method="highs")
        zs.append((-r.fun) if r.success else np.nan)
    st.plotly_chart(px.line(pd.DataFrame({"B": Bs, "Z*": zs}), x="B", y="Z*", markers=True, title="Độ nhạy theo ngân sách"), use_container_width=True)
    return {"z_lp": float(z)}


def page3() -> Dict[str, float]:
    st.title("Bài 3. Chỉ số ưu tiên ngành Priorityᵢ")
    df = load_sectors().copy()
    cols_good = ["growth_rate_2024_pct", "gdp_share_2024_pct", "spillover_coef_0_1", "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    labels = ["Tăng trưởng", "Năng suất/GDP share", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI readiness"]
    st.sidebar.subheader("Trọng số Bài 3")
    default_w = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20]
    weights = np.array([st.sidebar.slider(f"w · {name}", 0.0, 0.5, default_w[i], 0.01, key=f"p3w{i}") for i, name in enumerate(labels)])
    w_risk = st.sidebar.slider("w · Trừ rủi ro", 0.0, 0.5, 0.15, 0.01)

    Xg = df[cols_good].apply(minmax_good)
    risk_good = minmax_bad(df["automation_risk_pct"])
    priority = Xg.values @ weights + w_risk * risk_good.values
    df["Priority"] = priority
    result = df.sort_values("Priority", ascending=False)[["sector_name_vi", "Priority", "growth_rate_2024_pct", "ai_readiness_0_100", "automation_risk_pct"]]
    st.dataframe(result, use_container_width=True)
    st.plotly_chart(px.bar(result, x="Priority", y="sector_name_vi", orientation="h", title="Xếp hạng ưu tiên ngành"), use_container_width=True)

    sens = []
    for w_ai in np.arange(0.05, 0.45, 0.05):
        w = weights.copy(); w[5] = w_ai; w = w / w.sum() if w.sum() else weights
        p = Xg.values @ w + w_risk * risk_good.values
        top = df.assign(P=p).sort_values("P", ascending=False).head(3)["sector_name_vi"].tolist()
        sens.append({"w_AI": round(w_ai, 2), "Top 1": top[0], "Top 2": top[1], "Top 3": top[2]})
    st.subheader("Độ nhạy Top-3 theo trọng số AI readiness")
    st.dataframe(pd.DataFrame(sens), use_container_width=True)
    return {"top_priority": float(result.iloc[0]["Priority"])}


def solve_bai4(use_fairness: bool, budget: float) -> Tuple[pd.DataFrame, float]:
    regions = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items = ["I", "D", "AI", "H"]
    beta = np.array([
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ])
    if pulp is None:
        # Greedy fallback
        X = np.zeros((6, 4)); X[:, :] = 0
        X[:, 3] = 2000
        remaining = budget - X.sum()
        flat = np.dstack(np.unravel_index(np.argsort(-beta.ravel()), beta.shape))[0]
        region_sum = X.sum(axis=1)
        for r, j in flat:
            add = min(12000 - region_sum[r], remaining)
            if add > 0:
                X[r, j] += add; region_sum[r] += add; remaining -= add
            if remaining <= 0: break
        z = float((X * beta).sum())
    else:
        m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (range(6), range(4)), lowBound=0)
        m += pulp.lpSum(beta[r, j] * x[r][j] for r in range(6) for j in range(4))
        m += pulp.lpSum(x[r][j] for r in range(6) for j in range(4)) <= budget
        for r in range(6):
            m += pulp.lpSum(x[r][j] for j in range(4)) >= 5000
            m += pulp.lpSum(x[r][j] for j in range(4)) <= 12000
        m += pulp.lpSum(x[r][3] for r in range(6)) >= 0.24 * budget
        if use_fairness:
            D0 = [38, 78, 55, 32, 82, 48]
            gamma, lam = 0.002, 0.7
            M = pulp.LpVariable("Dmax", lowBound=0)
            for r in range(6):
                m += D0[r] + gamma * x[r][1] <= M
                m += D0[r] + gamma * x[r][1] >= lam * M
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        X = np.array([[pulp.value(x[r][j]) or 0 for j in range(4)] for r in range(6)])
        z = float(pulp.value(m.objective) or 0)
    table = pd.DataFrame(X, columns=items, index=regions).reset_index().rename(columns={"index": "Vùng"})
    return table, z


def page4() -> Dict[str, float]:
    st.title("Bài 4. LP phân bổ ngân sách số theo ngành-vùng")
    budget = st.slider("Ngân sách tổng", 30000, 72000, 50000, 1000)
    fairness = st.checkbox("Bật ràng buộc công bằng vùng miền C5", value=True)
    table, z = solve_bai4(fairness, budget)
    st.metric("GDP gain Z*", f"{z:,.1f}")
    st.dataframe(table, use_container_width=True)
    long = table.melt(id_vars="Vùng", var_name="Hạng mục", value_name="Ngân sách")
    st.plotly_chart(px.density_heatmap(long, x="Hạng mục", y="Vùng", z="Ngân sách", title="Heatmap phân bổ tối ưu"), use_container_width=True)
    table_no, z_no = solve_bai4(False, budget)
    st.info(f"Chi phí kinh tế của ràng buộc công bằng so với không công bằng: **{z_no - z:,.1f}** GDP gain.")
    return {"z_region": float(z)}


def page5() -> Dict[str, float]:
    st.title("Bài 5. MIP lựa chọn dự án chuyển đổi số")
    names = {
        1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc", 4: "VNeID 2.0", 5: "Dịch vụ công v3",
        6: "Y tế số", 7: "Giáo dục số K-12", 8: "Trung tâm AI", 9: "Sandbox fintech", 10: "Logistics thông minh",
        11: "Nông nghiệp số ĐBSCL", 12: "50.000 kỹ sư AI/bán dẫn", 13: "KCN bán dẫn", 14: "SOC an ninh mạng", 15: "Open Data",
    }
    C = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    budget = st.slider("Ngân sách 5 năm", 60000, 110000, 80000, 5000)
    force_p1p2 = st.checkbox("Bắt buộc chọn cả P1 và P2", value=False)
    if pulp is None:
        st.error("Thiếu PuLP để giải MIP.")
        return {}
    P = list(range(1,16)); m = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize); y = pulp.LpVariable.dicts("y", P, cat="Binary")
    m += pulp.lpSum(B[i]*y[i] for i in P)
    m += pulp.lpSum(C[i]*y[i] for i in P) <= budget
    m += pulp.lpSum(C1[i]*y[i] for i in P) <= 40000
    if not force_p1p2:
        m += y[1] + y[2] <= 1
    else:
        m += y[1] == 1; m += y[2] == 1
    m += y[8] <= y[12]; m += y[13] <= y[12]; m += y[4] + y[5] >= 1; m += y[14] >= 1
    m += pulp.lpSum(y[i] for i in P) >= 7; m += pulp.lpSum(y[i] for i in P) <= 11
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status]
    if status != "Optimal":
        st.error(f"Trạng thái: {status}. Bài toán có thể không khả thi.")
        return {}
    selected = [i for i in P if (pulp.value(y[i]) or 0) > 0.5]
    result = pd.DataFrame([{"Mã": f"P{i}", "Dự án": names[i], "Chi phí": C[i], "Lợi ích NPV": B[i], "Năm 1-2": C1[i]} for i in selected])
    st.metric("Tổng lợi ích NPV", f"{sum(B[i] for i in selected):,.0f}", f"{len(selected)} dự án")
    st.metric("Tổng chi phí", f"{sum(C[i] for i in selected):,.0f}")
    st.dataframe(result, use_container_width=True)
    st.plotly_chart(px.bar(result, x="Mã", y=["Chi phí", "Lợi ích NPV"], barmode="group", title="Chi phí và lợi ích dự án được chọn"), use_container_width=True)
    return {"npv_projects": float(sum(B[i] for i in selected))}


def page6() -> Dict[str, float]:
    st.title("Bài 6. TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI")
    df = load_regions().copy()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
    is_benefit = [True, True, True, True, True, True, True, False]
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    method = st.radio("Trọng số", ["Chuyên gia", "Entropy khách quan"], horizontal=True)
    X = df[criteria].values.astype(float)
    w = w_expert if method == "Chuyên gia" else entropy_weights(X)
    score = topsis(X, w, is_benefit)
    out = df.assign(TOPSIS_score=score).sort_values("TOPSIS_score", ascending=False)[["region_name_vi", "TOPSIS_score", "digital_index_0_100", "ai_readiness_0_100", "gini_coef"]]
    st.dataframe(out, use_container_width=True)
    st.plotly_chart(px.bar(out, x="TOPSIS_score", y="region_name_vi", orientation="h", title="Xếp hạng TOPSIS"), use_container_width=True)
    sens = []
    for w_ai in np.arange(0.10, 0.45, 0.05):
        ww = w_expert.copy(); ww[3] = w_ai; ww = ww / ww.sum()
        sc = topsis(X, ww, is_benefit)
        top3 = df.assign(sc=sc).sort_values("sc", ascending=False).head(3)["region_name_vi"].tolist()
        sens.append({"w_AI": round(w_ai,2), "Top-3": " | ".join(top3)})
    st.subheader("Độ nhạy Top-3 theo trọng số AI")
    st.dataframe(pd.DataFrame(sens), use_container_width=True)
    return {"topsis_best": float(out.iloc[0]["TOPSIS_score"])}


def page7() -> Dict[str, float]:
    st.title("Bài 7. Tối ưu đa mục tiêu Pareto với NSGA-II")
    st.caption("Bản Streamlit demo dùng lấy mẫu ngẫu nhiên + lọc Pareto để chạy nhẹ trên cloud.")
    n = st.slider("Số phương án mô phỏng", 500, 5000, 1500, 500)
    rng = np.random.default_rng(st.number_input("Seed", value=42, step=1))
    beta = np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
    e = np.array([0.42,0.55,0.48,0.32,0.62,0.38]); rho = np.array([0.18,0.45,0.28,0.12,0.52,0.22]); sig = np.array([0.32,0.28,0.30,0.35,0.25,0.30])
    Xs = rng.dirichlet(np.ones(24), size=n) * 50000
    F = []
    for x in Xs:
        X = x.reshape(6,4)
        f1 = (beta*X).sum()
        sums = X.sum(axis=1); f2 = np.abs(sums - sums.mean()).mean()
        f3 = (e * (X[:,0] + X[:,2])).sum()
        f4 = (rho * X[:,2]).sum() - (sig * X[:,3]).sum()
        F.append([f1, f2, f3, f4])
    F = np.array(F)
    # Pareto: maximize f1, minimize others -> transform to minimization
    G = np.c_[-F[:,0], F[:,1], F[:,2], F[:,3]]
    is_eff = np.ones(n, dtype=bool)
    for i, g in enumerate(G):
        if is_eff[i]:
            is_eff[is_eff] = np.any(G[is_eff] < g, axis=1) | np.all(G[is_eff] == g, axis=1)
            is_eff[i] = True
    pareto = pd.DataFrame(F[is_eff], columns=["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro ròng"])
    st.metric("Số nghiệm Pareto", len(pareto))
    fig = px.scatter_3d(pareto, x="GDP gain", y="Bất bình đẳng", z="Phát thải", color="Rủi ro ròng", title="Biên Pareto 3D")
    st.plotly_chart(fig, use_container_width=True)
    # TOPSIS compromise over Pareto
    vals = pareto.values
    sc = topsis(vals, np.array([0.40,0.25,0.20,0.15]), [True, False, False, False])
    best_idx = int(np.argmax(sc)); best = pareto.iloc[best_idx]
    metric_row([("GDP gain thỏa hiệp", f"{best['GDP gain']:,.0f}", None), ("Bất bình đẳng", f"{best['Bất bình đẳng']:,.0f}", None), ("Phát thải", f"{best['Phát thải']:,.0f}", None)])
    return {"pareto_gdp": float(best["GDP gain"])}


def page8() -> Dict[str, float]:
    st.title("Bài 8. Tối ưu động phân bổ liên thời gian 2026-2035")
    strategy = st.radio("Chiến lược", ["Đầu tư trải đều", "Front-load", "AI-H tăng dần"], horizontal=True)
    T = 10; years = np.arange(2026, 2036)
    K = np.zeros(T+1); D = np.zeros(T+1); AI = np.zeros(T+1); H = np.zeros(T+1); A = np.zeros(T+1)
    K[0]=27500; D[0]=20.3; AI[0]=86; H[0]=30; A[0]=1
    budget = st.slider("Ngân sách đầu tư hàng năm", 500, 2500, 1000, 100)
    C=[]; Y=[]
    for t in range(T):
        if strategy == "Đầu tư trải đều": alloc = np.array([0.35,0.25,0.20,0.20])
        elif strategy == "Front-load": alloc = np.array([0.45,0.30,0.15,0.10]) if t < 3 else np.array([0.25,0.20,0.20,0.35])
        else: alloc = np.array([0.30,0.20+0.02*t,0.15+0.02*t,0.35-0.04*t])
        alloc = np.maximum(alloc, 0.02); alloc = alloc/alloc.sum()
        Yt = A[t]*(K[t]**0.33)*(54**0.42)*(D[t]**0.10)*(AI[t]**0.08)*(H[t]**0.07)
        invest = budget
        C.append(max(Yt-invest, 1)); Y.append(Yt)
        K[t+1]=(1-0.05)*K[t]+alloc[0]*invest
        D[t+1]=(1-0.12)*D[t]+alloc[1]*invest/100
        AI[t+1]=(1-0.15)*AI[t]+alloc[2]*invest/20
        H[t+1]=H[t]+0.8*alloc[3]*invest/200-0.02*H[t]
        A[t+1]=A[t]*(1+0.003*D[t]/100+0.002*AI[t]/100+0.004*H[t]/100)
    welfare = sum((0.97**t)*math.log(c) for t,c in enumerate(C))
    df = pd.DataFrame({"Năm": years, "Y": Y, "C": C, "K": K[:-1], "D": D[:-1], "AI": AI[:-1], "H": H[:-1]})
    st.metric("Tổng welfare", f"{welfare:.2f}")
    st.plotly_chart(px.line(df, x="Năm", y=["Y", "C"], markers=True, title="Sản lượng và tiêu dùng"), use_container_width=True)
    st.plotly_chart(px.line(df, x="Năm", y=["K", "D", "AI", "H"], markers=True, title="Quỹ đạo trạng thái"), use_container_width=True)
    return {"welfare_dynamic": float(welfare), "Y2035": float(Y[-1])}


def page9() -> Dict[str, float]:
    st.title("Bài 9. Tác động AI tới thị trường lao động")
    sectors = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ", "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
    labor = np.array([13.2,11.5,4.8,7.8,0.55,1.95,0.62,2.15])
    risk = np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5]); b1=np.array([45,28,35,32,22,30,20,55]); c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5]); d1=np.array([50,32,42,38,26,36,24,62])
    budget = st.slider("Ngân sách tổng", 10000, 50000, 30000, 1000)
    if linprog is None:
        st.error("Thiếu scipy."); return {}
    # Variables: xAI_i, xH_i. maximize sum((a1-c1*risk)xAI + b1*xH)
    profit_ai = a1 - c1*risk
    c = -np.r_[profit_ai, b1]
    A_ub = [np.r_[np.ones(8), np.ones(8)]]; b_ub = [budget]
    # NetJob >=0 => -(profit_ai*xAI + b1*xH)<=0 ; Displaced <= retrain => c1*risk*xAI - d1*xH <=0
    for i in range(8):
        row = np.zeros(16); row[i] = -profit_ai[i]; row[8+i] = -b1[i]; A_ub.append(row); b_ub.append(0)
        row = np.zeros(16); row[i] = c1[i]*risk[i]; row[8+i] = -d1[i]; A_ub.append(row); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(0,None)]*16, method="highs")
    x = res.x if res.success else np.zeros(16)
    xAI, xH = x[:8], x[8:]
    new = a1*xAI; upgrade=b1*xH; displaced=c1*risk*xAI; net=new+upgrade-displaced
    out=pd.DataFrame({"Ngành":sectors,"Lao động triệu":labor,"x_AI":xAI,"x_H":xH,"NewJob":new,"Upgrade":upgrade,"Displaced":displaced,"NetJob":net})
    st.metric("Tổng NetJob", f"{net.sum():,.0f} việc làm")
    st.dataframe(out, use_container_width=True)
    st.plotly_chart(px.bar(out, x="Ngành", y=["NewJob", "Upgrade", "Displaced", "NetJob"], barmode="group", title="Tác động việc làm"), use_container_width=True)
    return {"net_jobs": float(net.sum())}


def page10() -> Dict[str, float]:
    st.title("Bài 10. Quy hoạch ngẫu nhiên hai giai đoạn")
    J = ["I", "D", "AI", "H"]
    p = np.array([0.30,0.45,0.20,0.05])
    S = ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"]
    beta = np.array([1.00,1.10,1.25,0.95])
    beta_s = np.array([[1.25,1.35,1.55,1.05],[1.00,1.10,1.25,0.95],[0.75,0.85,0.90,1.00],[0.40,0.50,0.55,1.10]])
    if pulp is None:
        st.error("Thiếu PuLP."); return {}
    m = pulp.LpProblem("SP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(4), lowBound=0); y = pulp.LpVariable.dicts("y", (range(4), range(4)), lowBound=0)
    m += pulp.lpSum(beta[j]*x[j] for j in range(4)) + pulp.lpSum(p[s]*pulp.lpSum(beta_s[s,j]*y[s][j] for j in range(4)) for s in range(4))
    m += pulp.lpSum(x[j] for j in range(4)) <= 65000
    for s in range(4):
        m += pulp.lpSum(y[s][j] for j in range(4)) <= 15000
        m += y[s][2] <= 0.5*x[3]
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    xval=np.array([pulp.value(x[j]) or 0 for j in range(4)]); yval=np.array([[pulp.value(y[s][j]) or 0 for j in range(4)] for s in range(4)])
    st.metric("Giá trị kỳ vọng SP", f"{pulp.value(m.objective):,.1f}")
    st.subheader("Quyết định first-stage")
    st.dataframe(pd.DataFrame({"Hạng mục":J,"x":xval}), use_container_width=True)
    st.subheader("Quyết định second-stage theo kịch bản")
    st.dataframe(pd.DataFrame(yval, columns=J, index=S), use_container_width=True)
    st.plotly_chart(px.bar(pd.DataFrame({"Hạng mục":J,"x":xval}), x="Hạng mục", y="x", title="First-stage allocation"), use_container_width=True)
    return {"sp_value": float(pulp.value(m.objective) or 0)}


def page11() -> Dict[str, float]:
    st.title("Bài 11. Học tăng cường Q-learning")
    episodes = st.slider("Số episode", 500, 10000, 3000, 500)
    lr = st.slider("Learning rate α", 0.01, 0.50, 0.10, 0.01)
    discount = st.slider("Discount γ", 0.50, 0.99, 0.95, 0.01)
    seed = st.number_input("Seed", value=42, step=1)
    rng = np.random.default_rng(seed)
    actions = {
        0: "Truyền thống", 1: "Cân bằng", 2: "Số hóa nhanh", 3: "AI dẫn dắt", 4: "Bao trùm"
    }
    Q = np.zeros((3,3,3,3,5))
    rewards=[]
    def step(state, action):
        g,d,ai,u = state
        a = action
        reward = 0.4*(g-1) + 0.2*(d-1) + 0.2*(ai-1) - 0.25*(u-1)
        if a == 2: d = min(2, d+1); reward += 0.35
        if a == 3: ai = min(2, ai+1); u = min(2, u+1); reward += 0.25
        if a == 4: u = max(0, u-1); reward += 0.30
        if a == 1: g = min(2, g+1); reward += 0.20
        if a == 0: reward += 0.05
        if rng.random() < 0.08: g = max(0, g-1)
        return np.array([g,d,ai,u]), reward
    if st.button("Huấn luyện Q-learning"):
        for ep in range(episodes):
            s=np.array([1,1,0,1]); total=0
            eps=max(0.05,1-ep/(episodes*0.6))
            for _ in range(10):
                a = rng.integers(5) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
                s2,r = step(s,a)
                Q[tuple(s)+(a,)] += lr*(r + discount*Q[tuple(s2)].max() - Q[tuple(s)+(a,)])
                total += r; s=s2
            rewards.append(total)
        curve=pd.DataFrame({"Episode":np.arange(len(rewards)),"Reward":rewards})
        curve["Reward mượt"] = curve["Reward"].rolling(50, min_periods=1).mean()
        st.session_state["q_curve"] = curve
        st.session_state["q_best_action"] = actions[int(np.argmax(Q[(1,1,0,1)]))]
    if "q_curve" in st.session_state:
        st.metric("Chính sách tại trạng thái VN 2026", st.session_state["q_best_action"])
        st.plotly_chart(px.line(st.session_state["q_curve"], x="Episode", y=["Reward", "Reward mượt"], title="Learning curve"), use_container_width=True)
        return {"q_reward": float(st.session_state["q_curve"]["Reward mượt"].iloc[-1])}
    st.info("Bấm huấn luyện để sinh learning curve.")
    return {}


def page12() -> Dict[str, float]:
    st.title("Bài 12. Dashboard tổng hợp AIDEOM-VN")
    st.caption("M6 · Tích hợp 6 module và so sánh 5 kịch bản chính sách")
    tabs = st.tabs(["Tổng quan", "Phân bổ", "Kịch bản so sánh", "Cảnh báo rủi ro", "Xuất báo cáo"])
    with tabs[0]:
        st.subheader("Pipeline 6 module")
        modules = pd.DataFrame([
            ["M1", "Dự báo kinh tế", "Macro 2020-2025", "GDP, TFP", "Cobb-Douglas"],
            ["M2", "Sẵn sàng số", "Sectors, Regions", "Digital + AI index", "TOPSIS"],
            ["M3", "Tối ưu phân bổ", "Budget, β-matrix", "Phân bổ ngành-vùng", "LP/Dynamic"],
            ["M4", "Lao động", "AI/H plan", "NetJob", "LP"],
            ["M5", "Rủi ro", "Risk parameters", "Cyber/Env", "Pareto/SP"],
            ["M6", "Dashboard", "Output M1-M5", "Kịch bản, cảnh báo", "Streamlit"],
        ], columns=["Module", "Tên", "Đầu vào", "Đầu ra", "Kỹ thuật"])
        st.dataframe(modules, use_container_width=True)
    with tabs[1]:
        table,z = solve_bai4(True, 50000)
        st.metric("Z* phân bổ vùng", f"{z:,.0f}")
        st.dataframe(table, use_container_width=True)
    with tabs[2]:
        scenarios = pd.DataFrame({
            "Kịch bản": ["S1 Truyền thống", "S2 Số hóa nhanh", "S3 AI dẫn dắt", "S4 Bao trùm", "S5 Tối ưu cân bằng"],
            "GDP_2030": [11.2,13.5,13.1,12.8,14.8],
            "NetJob": [80000,140000,110000,180000,165000],
            "Risk": [0.42,0.36,0.58,0.29,0.31],
        })
        st.dataframe(scenarios, use_container_width=True)
        st.plotly_chart(px.bar(scenarios, x="Kịch bản", y="GDP_2030", title="So sánh GDP 2030 theo kịch bản"), use_container_width=True)
        st.plotly_chart(px.scatter(scenarios, x="GDP_2030", y="NetJob", size="Risk", color="Kịch bản", title="Đánh đổi GDP - việc làm - rủi ro"), use_container_width=True)
    with tabs[3]:
        st.warning("Cảnh báo 1: Kịch bản AI dẫn dắt có rủi ro cao nếu thiếu đầu tư nhân lực và an ninh dữ liệu.")
        st.info("Cảnh báo 2: Kịch bản truyền thống cho tăng trưởng thấp nhất, không phù hợp mục tiêu kinh tế số.")
        st.success("Khuyến nghị: S5 tối ưu cân bằng là phương án thỏa hiệp tốt giữa GDP, NetJob và rủi ro.")
    with tabs[4]:
        st.download_button("Tải README mẫu", data=(APP_DIR/"README.md").read_text(encoding="utf-8"), file_name="README.md")
        st.caption("Có thể bổ sung xuất Excel/PDF từ outputs nếu cần.")
    return {"dashboard": 1.0}


def main():
    render_header()
    page = st.sidebar.radio("Chọn bài", PAGES, index=0)
    st.sidebar.divider()
    st.sidebar.caption("Dữ liệu: CSV trong thư mục data/. Có thể đẩy toàn bộ lên GitHub và deploy Streamlit Cloud.")
    mapping = {
        PAGES[0]: page1, PAGES[1]: page2, PAGES[2]: page3, PAGES[3]: page4, PAGES[4]: page5, PAGES[5]: page6,
        PAGES[6]: page7, PAGES[7]: page8, PAGES[8]: page9, PAGES[9]: page10, PAGES[10]: page11, PAGES[11]: page12,
    }
    mapping[page]()


if __name__ == "__main__":
    main()
