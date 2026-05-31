"""Bài 10: Two-stage stochastic programming via deterministic equivalent LP."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog
J=["I","D","AI","H"]
base=np.array([1.00,1.10,1.25,0.95])
BETA_S=np.array([[1.25,1.35,1.55,1.05],[1.00,1.10,1.25,0.95],[0.75,0.85,0.90,1.00],[0.40,0.50,0.55,1.10]])
S=["Lạc quan","Cơ sở","Bi quan","Khủng hoảng"]
DEFAULT_P=np.array([0.30,0.45,0.20,0.05])

def solve_sp(prob=None):
    p=np.asarray(prob if prob is not None else DEFAULT_P,float); p=p/(p.sum() or 1)
    # variables x4 + y(4 scenarios *4) =20
    n=20
    c=np.zeros(n); c[:4]=-base
    for s in range(4): c[4+4*s:4+4*(s+1)] = -p[s]*BETA_S[s]
    A=[]; b=[]
    row=np.zeros(n); row[:4]=1; A.append(row); b.append(65000)
    for s in range(4):
        row=np.zeros(n); row[4+4*s:4+4*(s+1)]=1; A.append(row); b.append(15000)
        row=np.zeros(n); row[4+4*s+2]=1; row[3] = -0.5; A.append(row); b.append(0)
    res=linprog(c,A_ub=np.array(A),b_ub=np.array(b),bounds=[(0,None)]*n,method="highs")
    x=res.x[:4]; y=res.x[4:].reshape(4,4); z=-res.fun
    xdf=pd.DataFrame({"Hạng mục":J,"First-stage x":x})
    ydf=pd.DataFrame(y, columns=J); ydf.insert(0,"Kịch bản",S)
    # Expected value solution: use average second beta, solve as deterministic with same constraints.
    ev_beta=(p[:,None]*BETA_S).sum(axis=0)
    # approximate EEV by evaluating base first-stage all into best expected item.
    z_ws=0
    for s in range(4):
        z_ws += p[s]*(65000*base.max()+15000*BETA_S[s].max())
    # EEV: deterministic solve evaluated in stochastic recourse
    c_ev=np.r_[-base, -ev_beta]
    Aev=np.zeros((2,8)); Aev[0,:4]=1; Aev[1,4:]=1
    rev=linprog(c_ev,A_ub=Aev,b_ub=[65000,15000],bounds=[(0,None)]*8,method="highs")
    xev=rev.x[:4]
    eev=(base*xev).sum()+sum(p[s]*15000*BETA_S[s].max() for s in range(4))
    metrics={"Z_SP":z,"EEV":eev,"WS":z_ws,"VSS":z-eev,"EVPI":z_ws-z}
    return xdf, ydf, metrics, res.success

def fig_x(xdf):
    return px.bar(xdf,x="Hạng mục",y="First-stage x",template="plotly_dark",title="Quyết định first-stage tối ưu")

def fig_metrics(metrics):
    df=pd.DataFrame({"Chỉ số":list(metrics),"Giá trị":list(metrics.values())})
    return px.bar(df,x="Chỉ số",y="Giá trị",template="plotly_dark",title="SP metrics: Z, EEV, WS, VSS, EVPI")
