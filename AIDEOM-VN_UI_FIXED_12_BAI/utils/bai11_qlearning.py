"""Bài 11: Tabular Q-learning for adaptive economic policy."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
ACTIONS = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
ALLOC = np.array([[0.70,0.10,0.10,0.10],[0.40,0.25,0.15,0.20],[0.25,0.45,0.15,0.15],[0.20,0.20,0.45,0.15],[0.30,0.20,0.10,0.40]])

def _step(state, action, rng):
    g,d,ai,u = state
    a = ALLOC[action]
    dgdp = 0.8*a[0] + 1.4*a[1] + 1.7*a[2] + 1.1*a[3] + 0.2*d + 0.15*ai - 0.15*u + rng.normal(0,0.05)
    du = 0.9*a[2] - 0.75*a[3] + 0.25*u + rng.normal(0,0.04)
    cyber = 0.55*a[2] + 0.18*ai
    emission = 0.45*a[0] + 0.35*a[2]
    reward = 0.40*dgdp - 0.25*du - 0.20*cyber - 0.15*emission
    g2 = int(np.clip(g + (dgdp>1.0) - (dgdp<0.55),0,2))
    d2 = int(np.clip(d + (a[1]>0.30) + (a[2]>0.30) - (a[0]>0.60),0,2))
    ai2 = int(np.clip(ai + (a[2]>0.30) + (d>=1 and a[1]>0.20) - (u==2 and a[3]<0.20),0,2))
    u2 = int(np.clip(u + (du>0.18) - (du<0.0),0,2))
    return np.array([g2,d2,ai2,u2]), reward

@__import__('streamlit').cache_data(show_spinner=False)
def train_q(episodes=8000, alpha=0.1, gamma=0.95, seed=42):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3,3,3,3,5))
    rewards=[]
    for ep in range(int(episodes)):
        s=np.array([1,1,0,1])
        total=0
        eps=max(0.05,1-ep/(episodes*0.65))
        for _ in range(10):
            if rng.random()<eps: a=int(rng.integers(5))
            else: a=int(np.argmax(Q[tuple(s)]))
            s2,r=_step(s,a,rng)
            Q[tuple(s)+(a,)] += alpha*(r + gamma*np.max(Q[tuple(s2)]) - Q[tuple(s)+(a,)])
            s=s2; total+=r
        rewards.append(total)
    curve=pd.DataFrame({"Episode":np.arange(len(rewards)),"Reward":rewards})
    curve["Smoothed"] = curve.Reward.rolling(100, min_periods=1).mean()
    states={"Việt Nam 2026":[1,1,0,1],"Tăng trưởng thấp - D thấp - U cao":[0,0,0,2],"GDP cao - AI cao - U thấp":[2,2,2,0],"Số hóa trung bình - rủi ro thất nghiệp cao":[1,1,1,2],"Nền tảng yếu":[0,0,1,1]}
    pol=[]
    for name,st in states.items():
        a=int(np.argmax(Q[tuple(st)])); pol.append({"Trạng thái":name,"Hành động tối ưu":ACTIONS[a],"Mã hành động":a})
    return curve, pd.DataFrame(pol), float(np.mean(rewards[-100:])), Q

def evaluate_rule(action=None, seed=1, episodes=300):
    rng=np.random.default_rng(seed); vals=[]
    for ep in range(episodes):
        s=np.array([1,1,0,1]); total=0
        for _ in range(10):
            a=int(rng.integers(5)) if action is None else action
            s,r=_step(s,a,rng); total+=r
        vals.append(total)
    return float(np.mean(vals))

def fig_curve(curve):
    long=curve.melt(id_vars="Episode", value_vars=["Reward","Smoothed"], var_name="Series", value_name="Reward value")
    return px.line(long,x="Episode",y="Reward value",color="Series",template="plotly_dark",title="Learning curve Q-learning")
