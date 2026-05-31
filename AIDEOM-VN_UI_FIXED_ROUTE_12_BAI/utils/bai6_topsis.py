"""Bài 6: TOPSIS region ranking and entropy weights."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from .data_loader import load_regions

CRITERIA = ["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100","ai_readiness_0_100","trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
BENEFIT = np.array([True,True,True,True,True,True,True,False])
DEFAULT_W = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])

def topsis(weights=None):
    df = load_regions().copy()
    w = np.asarray(weights if weights is not None else DEFAULT_W, dtype=float)
    w = w/(w.sum() or 1)
    X = df[CRITERIA].to_numpy(float)
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R*w
    A_star = np.where(BENEFIT, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(BENEFIT, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V-A_star)**2).sum(axis=1))
    S_neg = np.sqrt(((V-A_neg)**2).sum(axis=1))
    C = S_neg/(S_star+S_neg)
    out = df[["region_name_vi"]].copy(); out["TOPSIS_score"] = C; out["Rank"] = out.TOPSIS_score.rank(ascending=False, method="min").astype(int)
    return out.sort_values("TOPSIS_score", ascending=False), w

def entropy_weights():
    df = load_regions()
    X = df[CRITERIA].to_numpy(float).copy()
    # Convert cost criterion to benefit by inverse min-max before entropy.
    for j,b in enumerate(BENEFIT):
        if not b: X[:,j] = X[:,j].max() - X[:,j] + X[:,j].min()
    P = X / np.maximum(X.sum(axis=0), 1e-12)
    k = 1/np.log(len(X))
    E = -k*np.nansum(P*np.log(P+1e-12), axis=0)
    d = 1-E
    return d/d.sum()

def compare_methods(weights=None):
    a,_ = topsis(weights); a["Phương pháp"]="Chuyên gia"
    ew = entropy_weights(); b,_ = topsis(ew); b["Phương pháp"]="Entropy"
    return pd.concat([a,b]), ew

def fig_scores(df):
    return px.bar(df, x="TOPSIS_score", y="region_name_vi", color="Phương pháp", barmode="group", orientation="h", template="plotly_dark", title="So sánh TOPSIS chuyên gia vs Entropy")
