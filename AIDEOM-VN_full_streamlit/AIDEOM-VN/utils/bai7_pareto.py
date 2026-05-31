"""Bài 7: Multi-objective Pareto optimization with pymoo fallback."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from .bai4_lp_vung import BETA, REGIONS, ITEMS
E = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
RHO = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
SIG = np.array([0.32,0.28,0.30,0.35,0.25,0.30])

def _objectives(X):
    # X shape n,6,4, budget already around 50k
    f1 = (BETA*X).sum(axis=(1,2))
    sums = X.sum(axis=2)
    f2 = np.abs(sums - sums.mean(axis=1, keepdims=True)).mean(axis=1)
    f3 = (E*(X[:,:,0]+X[:,:,2])).sum(axis=1)
    f4 = (RHO*X[:,:,2]).sum(axis=1) - (SIG*X[:,:,3]).sum(axis=1)
    return np.c_[f1,f2,f3,f4]

@__import__('streamlit').cache_data(show_spinner=False)
def run_pareto(n_samples=400, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(24), size=n_samples)*50000
    X = X.reshape(n_samples,6,4)
    # enforce region floor/cap softly by mixing with equal floor
    X = 0.7*X + 0.3*np.ones((n_samples,6,4))*50000/24
    F = _objectives(X)
    # simple nondominated sort: max f1, min others
    G = np.c_[-F[:,0], F[:,1], F[:,2], F[:,3]]
    nd = np.ones(n_samples, dtype=bool)
    for i in range(n_samples):
        if not nd[i]: continue
        dominates_i = np.all(G <= G[i], axis=1) & np.any(G < G[i], axis=1)
        if dominates_i.any(): nd[i]=False
    pareto = pd.DataFrame(F[nd], columns=["GDP_gain","Inequality","Emission","Data_risk"])
    pareto["solution_id"] = np.arange(len(pareto))
    return pareto, X[nd]

def choose_topsis(pareto, weights=(0.40,0.25,0.20,0.15)):
    w = np.asarray(weights, float); w = w/(w.sum() or 1)
    X = pareto[["GDP_gain","Inequality","Emission","Data_risk"]].to_numpy(float)
    # benefit transform: GDP max, others min
    Z = X.copy(); Z[:,1:] = Z[:,1:].max(axis=0) - Z[:,1:] + Z[:,1:].min(axis=0)
    R = (Z-Z.min(axis=0))/(Z.max(axis=0)-Z.min(axis=0)+1e-9)
    score = R @ w
    out = pareto.copy(); out["Compromise_score"] = score
    best = out.loc[out.Compromise_score.idxmax()]
    high_growth = out.loc[out.GDP_gain.idxmax()]
    cost = {"GDP cao nhất":high_growth.GDP_gain,"GDP thỏa hiệp":best.GDP_gain,"Chênh lệch bất bình đẳng":high_growth.Inequality-best.Inequality,"Chênh lệch phát thải":high_growth.Emission-best.Emission}
    return out.sort_values("Compromise_score", ascending=False), best, cost

def fig_3d(pareto):
    return px.scatter_3d(pareto, x="GDP_gain", y="Inequality", z="Emission", color="Data_risk", hover_name="solution_id", template="plotly_dark", title="Đường biên Pareto 3D")

def fig_parallel(pareto):
    return px.parallel_coordinates(pareto[["GDP_gain","Inequality","Emission","Data_risk"]], color="GDP_gain", template="plotly_dark", title="Parallel coordinates 4 mục tiêu")
