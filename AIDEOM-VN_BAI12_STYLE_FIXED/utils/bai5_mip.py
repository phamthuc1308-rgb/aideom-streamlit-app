"""Bài 5: Mixed integer project selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
try:
    import pulp
except Exception:
    pulp = None

def project_table():
    names = {
        1:"Trung tâm dữ liệu quốc gia Hòa Lạc",2:"Trung tâm dữ liệu quốc gia phía Nam",3:"Hệ thống 5G phủ sóng toàn quốc",4:"VNeID 2.0",5:"Cổng dịch vụ công quốc gia v3",6:"Y tế số quốc gia",7:"Giáo dục số K-12",8:"Trung tâm AI quốc gia",9:"Sandbox fintech",10:"Logistics thông minh",11:"Nông nghiệp số ĐBSCL",12:"Đào tạo 50.000 kỹ sư AI/bán dẫn",13:"Khu CN bán dẫn Bắc Ninh - Bắc Giang",14:"An ninh mạng quốc gia",15:"Open Data quốc gia"
    }
    fields = {1:"Hạ tầng",2:"Hạ tầng",3:"Hạ tầng",4:"Chính phủ số",5:"Chính phủ số",6:"Y tế số",7:"Giáo dục",8:"AI",9:"Tài chính số",10:"Logistics",11:"Nông nghiệp",12:"Nhân lực",13:"Bán dẫn",14:"An ninh",15:"Dữ liệu"}
    C = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    df = pd.DataFrame([{"P":i,"Tên dự án":names[i],"Lĩnh vực":fields[i],"Chi phí":C[i],"Chi phí năm 1-2":C1[i],"NPV":B[i],"ROI":B[i]/C[i]} for i in range(1,16)])
    return df

def solve_mip(budget=80000, budget12=40000, min_projects=7, max_projects=11):
    df = project_table()
    if pulp is None:
        return df.assign(Chọn=False), np.nan, "PuLP chưa cài"
    m = pulp.LpProblem("bai5_mip", pulp.LpMaximize)
    P = df.P.tolist()
    y = pulp.LpVariable.dicts("y", P, cat="Binary")
    C = dict(zip(df.P, df["Chi phí"])); C1 = dict(zip(df.P, df["Chi phí năm 1-2"])); B = dict(zip(df.P, df.NPV))
    m += pulp.lpSum(B[i]*y[i] for i in P)
    m += pulp.lpSum(C[i]*y[i] for i in P) <= budget
    m += pulp.lpSum(C1[i]*y[i] for i in P) <= budget12
    m += y[1] + y[2] <= 1
    m += y[8] <= y[12]
    m += y[13] <= y[12]
    m += y[4] + y[5] >= 1
    m += y[14] >= 1
    m += pulp.lpSum(y[i] for i in P) >= min_projects
    m += pulp.lpSum(y[i] for i in P) <= max_projects
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    chosen = {i: bool(pulp.value(y[i]) > 0.5) for i in P}
    out = df.copy(); out["Chọn"] = out.P.map(chosen)
    return out, pulp.value(m.objective), pulp.LpStatus[m.status]

def fig_roi(df):
    return px.bar(df, x="P", y="ROI", color="Chọn", hover_data=["Tên dự án","Chi phí","NPV"], template="plotly_dark", title="ROI dự án và lựa chọn tối ưu")
