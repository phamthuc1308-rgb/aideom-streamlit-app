"""Bài 9: AI and labor market LP."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog
SECTORS = ["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo"]
risk = np.array([18,42,25,38,52,35,28,22])/100
a1 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
b1 = np.array([45,28,35,32,22,30,20,55])
c1 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
d1 = np.array([50,32,42,38,26,36,24,62])

def solve_labor(budget=30000):
    N=8
    # variables [xAI0..7, xH0..7]
    net_ai = a1 - c1*risk
    net_h = b1
    c = -np.r_[net_ai, net_h]
    A=[]; b=[]
    A.append(np.ones(2*N)); b.append(budget)
    # NetJob >= 0 => -net_ai*xAI - b1*xH <=0
    for i in range(N):
        row=np.zeros(2*N); row[i] = -net_ai[i]; row[N+i] = -b1[i]
        A.append(row); b.append(0)
    # Displaced <= retrain capacity: c1*risk*xAI - d1*xH <=0
    for i in range(N):
        row=np.zeros(2*N); row[i] = c1[i]*risk[i]; row[N+i] = -d1[i]
        A.append(row); b.append(0)
    res=linprog(c,A_ub=np.array(A),b_ub=np.array(b),bounds=[(0,None)]*(2*N),method="highs")
    x=res.x if res.success else np.zeros(2*N)
    xAI,xH=x[:N],x[N:]
    df=pd.DataFrame({"Ngành":SECTORS,"x_AI":xAI,"x_H":xH,"NewJob_AI":a1*xAI,"UpgradeJob":b1*xH,"Displaced":c1*risk*xAI})
    df["NetJob"] = df.NewJob_AI+df.UpgradeJob-df.Displaced
    return res, df, float(df.NetJob.sum())

def fig_alloc(df):
    long=df.melt(id_vars="Ngành", value_vars=["x_AI","x_H"], var_name="Đầu tư", value_name="Tỷ VND")
    return px.bar(long,x="Ngành",y="Tỷ VND",color="Đầu tư",template="plotly_dark",title="Phân bổ x_AI và x_H")

def fig_jobs(df):
    long=df.melt(id_vars="Ngành", value_vars=["NewJob_AI","UpgradeJob","Displaced","NetJob"], var_name="Thành phần", value_name="Việc làm")
    return px.bar(long,x="Ngành",y="Việc làm",color="Thành phần",barmode="group",template="plotly_dark",title="Phân rã việc làm")
