"""Bài 1: Extended Cobb-Douglas production model."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .data_loader import load_macro

DEFAULT_COEFS = dict(alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07)

def normalize_coefs(alpha: float, beta: float, gamma: float, delta: float, theta: float) -> dict:
    arr = np.array([alpha,beta,gamma,delta,theta], dtype=float)
    s = arr.sum() or 1.0
    arr = arr / s
    return dict(alpha=arr[0], beta=arr[1], gamma=arr[2], delta=arr[3], theta=arr[4])

def compute_model(alpha=0.33,beta=0.42,gamma=0.10,delta=0.08,theta=0.07):
    c = normalize_coefs(alpha,beta,gamma,delta,theta)
    df = load_macro().copy()
    Y = df["GDP_trillion_VND"].to_numpy(float)
    K = df["K_accumulated_trillion_VND"].to_numpy(float)
    L = df["labor_million"].to_numpy(float)
    D = df["digital_economy_share_GDP_pct"].to_numpy(float)
    AI = df["ai_firms_thousand"].to_numpy(float)
    H = df["trained_labor_pct"].to_numpy(float)
    A = Y/(K**c['alpha']*L**c['beta']*D**c['gamma']*AI**c['delta']*H**c['theta'])
    df["TFP_A"] = A
    Abar = float(np.mean(A))
    df["Y_hat"] = Abar*(K**c['alpha']*L**c['beta']*D**c['gamma']*AI**c['delta']*H**c['theta'])
    df["APE_pct"] = np.abs(df["Y_hat"]-df["GDP_trillion_VND"])/df["GDP_trillion_VND"]*100
    mape = float(df["APE_pct"].mean())
    log = lambda x: np.log(np.asarray(x, float))
    dlnY = log(Y[-1])-log(Y[0])
    contrib = {
        "K": c['alpha']*(log(K[-1])-log(K[0])),
        "L": c['beta']*(log(L[-1])-log(L[0])),
        "D": c['gamma']*(log(D[-1])-log(D[0])),
        "AI": c['delta']*(log(AI[-1])-log(AI[0])),
        "H": c['theta']*(log(H[-1])-log(H[0])),
        "TFP": (log(A[-1])-log(A[0])),
    }
    contrib_df = pd.DataFrame({"Yếu tố":list(contrib),"Đóng góp log":list(contrib.values())})
    contrib_df["Tỷ trọng trong tăng trưởng (%)"] = contrib_df["Đóng góp log"] / dlnY * 100
    years_to_2030 = 5
    K2030 = K[-1]*(1.06**years_to_2030)
    L2030 = L[-1]*(1.06**years_to_2030)
    A2030 = A[-1]*(1.012**years_to_2030)
    Y2030 = A2030*(K2030**c['alpha']*L2030**c['beta']*30**c['gamma']*100**c['delta']*35**c['theta'])
    return df, contrib_df, mape, float(Y2030), c

def fig_tfp(df):
    return px.line(df, x="year", y="TFP_A", markers=True, title="TFP A_t theo năm", template="plotly_dark")

def fig_actual_vs_pred(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.year, y=df.GDP_trillion_VND, mode="lines+markers", name="Y thực tế"))
    fig.add_trace(go.Scatter(x=df.year, y=df.Y_hat, mode="lines+markers", name="Ŷ dự báo"))
    fig.update_layout(template="plotly_dark", title="GDP thực tế vs dự báo", xaxis_title="Năm", yaxis_title="Nghìn tỷ VND")
    return fig

def fig_contrib(contrib_df):
    return px.bar(contrib_df, x="Yếu tố", y="Tỷ trọng trong tăng trưởng (%)", text_auto=".1f", template="plotly_dark", title="Phân rã tăng trưởng 2020–2025")
