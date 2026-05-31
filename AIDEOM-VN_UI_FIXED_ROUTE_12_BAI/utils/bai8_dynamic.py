"""Bài 8: Dynamic allocation 2026-2035 with SLSQP over investment shares."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

ALPHA,BETA,GAMMA,DELTA,THETA = 0.33,0.42,0.10,0.08,0.07

def simulate(shares, rho=0.97, dK=0.05, dD=0.12, dAI=0.15, T=10, frontload=False):
    shares = np.asarray(shares, float); shares = shares/(shares.sum() or 1)
    K,D,AI,H,A,L = 27500.0,20.3,86.0,30.0,1.0,53.9
    rows=[]; welfare=0.0
    for t in range(T):
        year = 2026+t
        Y = A*(K**ALPHA)*(L**BETA)*(D**GAMMA)*(AI**DELTA)*(H**THETA)
        inv_rate = 0.24
        if frontload: inv_rate = 0.32 if t < 3 else 0.18
        invest = inv_rate*Y
        C = max(Y-invest, 1e-6)
        IK,ID,IAI,IH = invest*shares
        welfare += (rho**t)*np.log(C)
        rows.append({"Năm":year,"K":K,"D":D,"AI":AI,"H":H,"A":A,"Y":Y,"C":C,"I_K":IK,"I_D":ID,"I_AI":IAI,"I_H":IH})
        K = (1-dK)*K + IK
        D = (1-dD)*D + ID/1000
        AI = (1-dAI)*AI + IAI/120
        H = H + 0.8*IH/1000 - 0.02*H
        A = A*(1 + 0.003*D/100 + 0.002*AI/100 + 0.004*H/100)
        L *= 1.006
    return pd.DataFrame(rows), welfare

def optimize_policy(rho=0.97,dK=0.05,dD=0.12,dAI=0.15):
    def obj(z):
        _, w = simulate(z, rho,dK,dD,dAI)
        return -w
    cons = ({"type":"eq","fun":lambda z: np.sum(z)-1})
    bounds = [(0.05,0.75)]*4
    res = minimize(obj, x0=np.array([0.35,0.25,0.20,0.20]), bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter":200})
    shares = res.x/(res.x.sum() or 1)
    df, w = simulate(shares, rho,dK,dD,dAI)
    return shares, df, w, res.success

def compare_strategies(rho=0.97,dK=0.05,dD=0.12,dAI=0.15):
    even_shares = np.array([0.25,0.25,0.25,0.25])
    opt_shares, opt_df, opt_w, _ = optimize_policy(rho,dK,dD,dAI)
    even_df, even_w = simulate(even_shares,rho,dK,dD,dAI)
    front_df, front_w = simulate(opt_shares,rho,dK,dD,dAI,frontload=True)
    return pd.DataFrame({"Chiến lược":["Tối ưu tỷ lệ","Đầu tư đều","Front-load"],"Welfare":[opt_w,even_w,front_w],"GDP 2035":[opt_df.Y.iloc[-1],even_df.Y.iloc[-1],front_df.Y.iloc[-1]]}), opt_shares, opt_df, front_df

def fig_trajectories(df):
    long = df.melt(id_vars="Năm", value_vars=["K","D","AI","H","Y","C"], var_name="Biến", value_name="Giá trị")
    return px.line(long, x="Năm", y="Giá trị", color="Biến", template="plotly_dark", title="Quỹ đạo tối ưu 2026–2035")
