"""Bài 3: Sector priority index and sensitivity analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from .data_loader import load_sectors

GOOD_COLS = ["growth_rate_2024_pct","gdp_share_2024_pct","spillover_coef_0_1","export_billion_USD","labor_million","ai_readiness_0_100"]
BAD_COL = "automation_risk_pct"
LABELS = ["Growth","Productivity","Spillover","Export","Employment","AI readiness"]

def _norm_good(s):
    den = s.max()-s.min()
    return (s-s.min())/(den if den else 1)

def _norm_bad(s):
    den = s.max()-s.min()
    return (s.max()-s)/(den if den else 1)

def normalized_matrix():
    df = load_sectors().copy()
    Xg = df[GOOD_COLS].apply(_norm_good)
    Xb = _norm_bad(df[BAD_COL])
    out = pd.concat([df[["sector_name_vi"]], Xg, Xb.rename("risk_good")], axis=1)
    return df, out

def compute_priority(weights=None, w_risk=0.15):
    if weights is None:
        weights = np.array([0.15,0.15,0.20,0.15,0.10,0.20])
    weights = np.asarray(weights, dtype=float)
    df, norm = normalized_matrix()
    score = norm[GOOD_COLS].to_numpy() @ weights - w_risk * norm["risk_good"].to_numpy()
    res = df[["sector_name_vi","sector_name_en"]].copy()
    res["Priority"] = score
    res["Rank"] = res["Priority"].rank(ascending=False, method="min").astype(int)
    return res.sort_values("Priority", ascending=False), norm

def sensitivity_ai():
    rows=[]
    base = np.array([0.15,0.15,0.20,0.15,0.10,0.20])
    for wai in np.arange(0.05,0.401,0.05):
        w = base.copy(); w[5] = wai; w = w / w.sum() * base.sum()
        rank, _ = compute_priority(w, 0.15)
        for _, r in rank.iterrows():
            rows.append({"AI weight":round(wai,2),"Ngành":r["sector_name_vi"],"Rank":r["Rank"],"Priority":r["Priority"]})
    return pd.DataFrame(rows)

def compare_weight_sets():
    growth = np.array([0.25,0.25,0.12,0.22,0.05,0.11])
    inclusive = np.array([0.10,0.10,0.25,0.05,0.25,0.10])
    a,_ = compute_priority(growth,0.10); a["Bộ trọng số"]="Tăng trưởng"
    b,_ = compute_priority(inclusive,0.25); b["Bộ trọng số"]="Bao trùm"
    return pd.concat([a,b])

def fig_rank(df):
    return px.bar(df, x="Priority", y="sector_name_vi", orientation="h", text="Rank", template="plotly_dark", title="Xếp hạng Priority Index").update_layout(yaxis={"categoryorder":"total ascending"})

def fig_heatmap(sens):
    pivot = sens.pivot(index="Ngành", columns="AI weight", values="Rank")
    return px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r", template="plotly_dark", title="Độ nhạy thứ hạng theo trọng số AI readiness")
