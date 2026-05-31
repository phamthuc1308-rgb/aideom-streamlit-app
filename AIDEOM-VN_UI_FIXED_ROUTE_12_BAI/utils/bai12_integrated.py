"""Bài 12: Integrated AIDEOM-VN scenario dashboard."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
SCENARIOS = {
    "S1 Truyền thống": np.array([0.70,0.10,0.10,0.10]),
    "S2 Số hóa nhanh": np.array([0.25,0.45,0.15,0.15]),
    "S3 AI dẫn dắt": np.array([0.20,0.20,0.45,0.15]),
    "S4 Bao trùm số": np.array([0.30,0.20,0.10,0.40]),
    "S5 Tối ưu cân bằng": np.array([0.35,0.28,0.20,0.17]),
}

def simulate_scenarios():
    rows=[]
    for name, sh in SCENARIOS.items():
        K,D,AI,H,A,L = 25900.0,19.5,80.1,29.2,1.0,53.4
        for year in range(2026,2031):
            Y = A*(K**0.33)*(L**0.42)*(D**0.10)*(AI**0.08)*(H**0.07)
            rows.append({"Kịch bản":name,"Năm":year,"GDP_index":Y,"K":K,"D":D,"AI":AI,"H":H})
            invest = 0.22*Y
            K = 0.95*K + invest*sh[0]
            D = 0.88*D + invest*sh[1]/1000
            AI = 0.85*AI + invest*sh[2]/120
            H = H + 0.8*invest*sh[3]/1000 - 0.02*H
            A = A*(1+0.012+0.002*D/100+0.002*AI/100)
            L *= 1.006
    df=pd.DataFrame(rows)
    kpi=df.sort_values("Năm").groupby("Kịch bản").tail(1).copy()
    kpi["Digital_score"] = kpi.D
    kpi["AI_score"] = kpi.AI
    kpi["Human_score"] = kpi.H
    kpi["Risk_control"] = 100 - (kpi.AI*0.2 - kpi.H*0.35)
    return df, kpi

def fig_gdp(df):
    return px.line(df,x="Năm",y="GDP_index",color="Kịch bản",markers=True,template="plotly_dark",title="Đường GDP mô phỏng 2026–2030 theo 5 kịch bản")

def fig_radar(kpi):
    metrics=["GDP_index","Digital_score","AI_score","Human_score","Risk_control"]
    norm=kpi.copy()
    for m in metrics:
        norm[m]=(norm[m]-norm[m].min())/(norm[m].max()-norm[m].min()+1e-9)
    fig=go.Figure()
    for _,r in norm.iterrows():
        fig.add_trace(go.Scatterpolar(r=[r[m] for m in metrics]+[r[metrics[0]]], theta=metrics+[metrics[0]], fill='toself', name=r["Kịch bản"]))
    fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Radar 5 trục KPI 2030")
    return fig
