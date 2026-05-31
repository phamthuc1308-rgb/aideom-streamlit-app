"""Bài 4: LP budget allocation across 6 regions and 4 investment items."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
try:
    import pulp
except Exception:
    pulp = None

REGIONS = ["NMM","RRD","NCC","CH","SE","MD"]
REGION_NAMES = ["Trung du miền núi phía Bắc","Đồng bằng sông Hồng","Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","Đồng bằng sông Cửu Long"]
ITEMS = ["I","D","AI","H"]
BETA = np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
D0 = np.array([38,78,55,32,82,48], dtype=float)

def solve_lp(lambda_fair=0.7, floor_region=5000, cap_region=12000, budget=50000, h_min=12000, use_fair=True):
    if pulp is None:
        return None, pd.DataFrame(), np.nan, "PuLP chưa được cài."
    m = pulp.LpProblem("bai4_region_lp", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(6), range(4)), lowBound=0)
    m += pulp.lpSum(BETA[r,j]*x[r][j] for r in range(6) for j in range(4))
    m += pulp.lpSum(x[r][j] for r in range(6) for j in range(4)) <= budget
    for r in range(6):
        m += pulp.lpSum(x[r][j] for j in range(4)) >= floor_region
        m += pulp.lpSum(x[r][j] for j in range(4)) <= cap_region
    m += pulp.lpSum(x[r][3] for r in range(6)) >= h_min
    if use_fair:
        M = pulp.LpVariable("Dmax", lowBound=0)
        for r in range(6):
            m += D0[r] + 0.002*x[r][1] <= M
        for r in range(6):
            m += D0[r] + 0.002*x[r][1] >= lambda_fair*M
    status = m.solve(pulp.PULP_CBC_CMD(msg=False))
    ok = pulp.LpStatus[m.status] == "Optimal"
    rows=[]
    if ok:
        for r in range(6):
            for j in range(4):
                rows.append({"Vùng":REGION_NAMES[r],"Mã vùng":REGIONS[r],"Hạng mục":ITEMS[j],"Phân bổ":pulp.value(x[r][j]),"Beta":BETA[r,j]})
    return m, pd.DataFrame(rows), (pulp.value(m.objective) if ok else np.nan), pulp.LpStatus[m.status]

def find_max_feasible_lambda(floor_region=5000, cap_region=12000):
    best=0
    for lam in np.linspace(0.1,1.0,91):
        _,_,_,status = solve_lp(lam, floor_region, cap_region, use_fair=True)
        if status == "Optimal": best = lam
    return best

def fairness_cost(lambda_fair=0.7, floor_region=5000, cap_region=12000):
    _,_,z_fair,st1 = solve_lp(lambda_fair,floor_region,cap_region,use_fair=True)
    _,_,z_free,st2 = solve_lp(lambda_fair,floor_region,cap_region,use_fair=False)
    return {"Z có C5":z_fair,"Z không C5":z_free,"Chi phí công bằng":z_free-z_fair,"% giảm":(z_free-z_fair)/z_free*100 if z_free else np.nan,"Trạng thái":st1}

def fig_heatmap(df):
    p = df.pivot(index="Vùng", columns="Hạng mục", values="Phân bổ")
    return px.imshow(p, text_auto=".0f", color_continuous_scale="Teal", template="plotly_dark", title="Heatmap phân bổ 6×4")

def fig_region_totals(df):
    totals = df.groupby("Vùng", as_index=False)["Phân bổ"].sum()
    return px.bar(totals, x="Vùng", y="Phân bổ", template="plotly_dark", title="Tổng ngân sách theo vùng")
