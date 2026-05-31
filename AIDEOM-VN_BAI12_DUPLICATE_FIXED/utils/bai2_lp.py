"""Bài 2: Linear programming budget allocation with SciPy and PuLP."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog
try:
    import pulp
except Exception:
    pulp = None

ITEMS = ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D công nghệ"]
OBJ = np.array([0.85,1.20,0.95,1.35])

def solve_scipy(budget=100, floors=(25,15,20,10), strategic_share=0.35):
    c = -OBJ
    A_ub = [[1,1,1,1]]
    b_ub = [budget]
    for i, floor in enumerate(floors):
        row = [0,0,0,0]; row[i] = -1
        A_ub.append(row); b_ub.append(-floor)
    # x2+x4 >= s*sum => s*x1 +(s-1)x2 + s*x3 +(s-1)x4 <= 0
    s = strategic_share
    A_ub.append([s, s-1, s, s-1]); b_ub.append(0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*4, method="highs")
    x = res.x if res.success else np.full(4, np.nan)
    df = pd.DataFrame({"Hạng mục":ITEMS,"Phân bổ":x,"Hệ số GDP":OBJ,"GDP gain":x*OBJ})
    return res, df, (-res.fun if res.success else np.nan)

def solve_pulp(budget=100, floors=(25,15,20,10), strategic_share=0.35):
    if pulp is None:
        return None, pd.DataFrame()
    m = pulp.LpProblem("bai2_lp", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(4), lowBound=0)
    m += pulp.lpSum(OBJ[i]*x[i] for i in range(4))
    cons = {}
    cons["Ngân sách"] = pulp.lpSum(x[i] for i in range(4)) <= budget
    for k,v in cons.items(): m += v, k
    for i,f in enumerate(floors):
        m += x[i] >= f, f"Sàn {ITEMS[i]}"
    s = strategic_share
    m += x[1]+x[3] >= s*pulp.lpSum(x[i] for i in range(4)), "Tỷ trọng công nghệ chiến lược"
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    shadow = []
    for name,c in m.constraints.items():
        shadow.append({"Ràng buộc":name,"Slack":getattr(c,"slack",np.nan),"Shadow price":getattr(c,"pi",np.nan)})
    return m, pd.DataFrame(shadow)

def sensitivity_curve(floors=(25,15,20,10)):
    rows=[]
    for B in range(100, 161, 10):
        res, _, z = solve_scipy(B, floors=floors)
        rows.append({"Ngân sách":B,"Z*":z,"Khả thi":res.success})
    return pd.DataFrame(rows)

def fig_allocation(df):
    return px.bar(df, x="Hạng mục", y="Phân bổ", color="Hạng mục", template="plotly_dark", title="Phân bổ tối ưu")

def fig_sensitivity(df):
    return px.line(df, x="Ngân sách", y="Z*", markers=True, template="plotly_dark", title="Đường cong Z*(B)")
