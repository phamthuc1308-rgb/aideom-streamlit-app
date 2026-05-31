import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
from itertools import product

st.set_page_config(page_title="AIDEOM-VN Interactive", page_icon="🇻🇳", layout="wide")

# -----------------------------
# THEME / STYLE
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');
:root{--navy:#111a33;--navy2:#172445;--pink:#d94d83;--violet:#7245d1;--gold:#e8b657;--teal:#28b8a2;--paper:#f7f2e8;--ink:#111827;}
html, body, [class*="css"]{font-family:Inter,system-ui,sans-serif;}
.stApp{background:linear-gradient(135deg,#0b1024 0%,#101932 40%,#11172b 100%);color:#edf2ff;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1360px;}
h1,h2,h3{font-weight:800;letter-spacing:-.03em;color:#f8fbff!important;}
p, li, label, .stMarkdown{color:#e6ecff;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d142b,#111b36);border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] *{color:#edf2ff!important;}
.hero{padding:34px 34px 28px;border:1px solid rgba(255,255,255,.12);border-radius:28px;background:radial-gradient(circle at 10% 10%,rgba(217,77,131,.35),transparent 35%),linear-gradient(135deg,rgba(255,255,255,.10),rgba(255,255,255,.035));box-shadow:0 18px 60px rgba(0,0,0,.28);margin-bottom:22px;}
.hero .eyebrow{display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:linear-gradient(90deg,var(--pink),var(--violet));font-size:12px;font-weight:700;margin-bottom:14px;}
.hero h1{font-size:46px;line-height:1.05;margin:0 0 10px;}
.hero p{font-size:16px;color:#cdd7f7;margin:0;max-width:950px;}
.card{padding:18px 20px;border:1px solid rgba(255,255,255,.10);border-radius:20px;background:rgba(255,255,255,.055);box-shadow:0 10px 35px rgba(0,0,0,.16);height:100%;}
.kpi{padding:18px 18px;border-radius:20px;background:linear-gradient(135deg,rgba(217,77,131,.20),rgba(114,69,209,.12));border:1px solid rgba(255,255,255,.12);}
.kpi .label{font-size:12px;color:#aebbe5;text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
.kpi .value{font-size:27px;font-weight:800;color:#fff;margin-top:4px;}
.kpi .note{font-size:12px;color:#cbd5ff;margin-top:4px;}
.badge{display:inline-flex;padding:6px 11px;border-radius:999px;background:rgba(232,182,87,.15);border:1px solid rgba(232,182,87,.35);color:#ffe1a1;font-weight:700;font-size:12px;margin-right:6px;margin-bottom:6px;}
.policy{border-left:4px solid var(--gold);padding:14px 16px;border-radius:14px;background:rgba(232,182,87,.08);margin-top:12px;color:#f7edd2;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{border-radius:999px;background:rgba(255,255,255,.06);padding:10px 16px;color:#dfe7ff;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,var(--pink),var(--violet));color:white;}
[data-testid="stMetricValue"]{color:white;}
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;}
hr{border-color:rgba(255,255,255,.10)}
.small{font-size:13px;color:#b8c2e8;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def macro_data():
    return pd.DataFrame({
        "year":[2020,2021,2022,2023,2024,2025],
        "GDP_trillion_VND":[8044.4,8487.5,9513.3,10221.8,11511.9,12847.6],
        "K_capital":[16500,17800,19600,21300,23500,25900],
        "L_labor_million":[53.6,50.5,51.7,52.4,52.9,53.4],
        "D_digital_pct":[12.0,12.7,14.3,16.5,18.3,19.5],
        "AI_firms_thousand":[55.6,60.2,65.4,67.0,73.8,80.1],
        "H_trained_labor_pct":[24.1,26.1,26.2,27.0,28.4,29.2]
    })

@st.cache_data
def sectors_data():
    names=["Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Khai khoáng","Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo","Y tế"]
    return pd.DataFrame({
        "sector":names,
        "growth":[3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85],
        "productivity":[103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1],
        "spillover":[0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
        "export":[40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
        "employment":[13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
        "ai_readiness":[15,55,20,30,48,72,42,88,38,45],
        "automation_risk":[18,42,25,55,38,52,35,28,22,18]
    })

@st.cache_data
def regions_data():
    return pd.DataFrame({
        "region":["Trung du miền núi phía Bắc","Đồng bằng sông Hồng","Bắc Trung Bộ + DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","Đồng bằng sông Cửu Long"],
        "code":["NMM","RRD","NCC","CH","SE","MD"],
        "grdp_pc":[57.0,152.3,87.5,68.9,158.9,80.5],
        "fdi":[3.5,20.0,8.2,0.8,18.5,2.1],
        "digital_index":[38,78,55,32,82,48],
        "ai_readiness":[22,68,40,18,75,30],
        "trained_labor":[21.5,36.8,27.5,18.2,42.5,16.8],
        "rd_intensity":[0.18,0.85,0.32,0.15,0.78,0.22],
        "internet":[72,92,84,68,94,78],
        "gini":[0.405,0.358,0.372,0.412,0.385,0.392]
    })

@st.cache_data
def projects_data():
    return pd.DataFrame({
        "id":[f"P{i}" for i in range(1,16)],
        "name":["Trung tâm dữ liệu Hòa Lạc","Trung tâm dữ liệu phía Nam","5G phủ sóng toàn quốc","VNeID 2.0","Dịch vụ công quốc gia v3","Y tế số quốc gia","Giáo dục số K-12","Trung tâm AI quốc gia","Sandbox fintech","Logistics thông minh","Nông nghiệp số ĐBSCL","Đào tạo 50.000 kỹ sư AI/bán dẫn","Khu CN bán dẫn Bắc Ninh - Bắc Giang","SOC an ninh mạng quốc gia","Open Data quốc gia"],
        "field":["Hạ tầng","Hạ tầng","Hạ tầng","Chính phủ số","Chính phủ số","Y tế","Giáo dục","AI","Tài chính số","Logistics","Nông nghiệp","Nhân lực","Bán dẫn","An ninh","Dữ liệu"],
        "cost":[12000,11500,18000,4500,3200,5800,6500,15000,2500,7200,4800,8500,20000,3800,1500],
        "benefit":[21500,20800,32500,9200,6800,11400,12200,28500,5800,13800,8500,16200,35000,7500,3800],
        "y12":[8500,7500,12000,3500,2500,4000,4500,9000,1800,5000,3500,5500,13000,2800,1200],
        "y35":[3500,4000,6000,1000,700,1800,2000,6000,700,2200,1300,3000,7000,1000,300]
    })

# -----------------------------
# HELPERS
# -----------------------------
def hero(title, subtitle, badges=None):
    badges_html="" if not badges else "".join([f"<span class='badge'>{b}</span>" for b in badges])
    st.markdown(f"""
    <div class='hero'>
      <div class='eyebrow'>🇻🇳 AIDEOM-VN Interactive Dashboard</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
      <div style='margin-top:14px'>{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)

def kpi(label, value, note=""):
    st.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div><div class='note'>{note}</div></div>", unsafe_allow_html=True)

def clean_fig(fig, height=440):
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.03)", margin=dict(l=20,r=20,t=55,b=20), font=dict(family="Inter", color="#e7ecff"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def norm_good(s):
    s=pd.Series(s,dtype=float)
    if s.max()==s.min(): return s*0
    return (s-s.min())/(s.max()-s.min())

def norm_bad(s):
    s=pd.Series(s,dtype=float)
    if s.max()==s.min(): return s*0
    return (s.max()-s)/(s.max()-s.min())

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown("### AIDEOM·VN")
st.sidebar.caption("Streamlit dashboard tương tác cho 12 bài")
page = st.sidebar.radio("Chọn phần", [
    "Trang chủ", "Bài 1 · Cobb-Douglas", "Bài 2 · LP ngân sách", "Bài 3 · Priority ngành",
    "Bài 4 · LP ngành-vùng", "Bài 5 · MIP dự án", "Bài 6 · TOPSIS vùng", "Bài 7 · Pareto NSGA-II",
    "Bài 8 · Dynamic Programming", "Bài 9 · AI & lao động", "Bài 10 · Stochastic LP", "Bài 11 · Q-learning", "Bài 12 · Dashboard tích hợp"
])
st.sidebar.divider()
st.sidebar.info("Gợi ý: dùng sidebar trong từng bài để thay tham số, chỉnh bảng dữ liệu và xem biểu đồ cập nhật tức thì.")

# -----------------------------
# HOME
# -----------------------------
if page == "Trang chủ":
    hero("Phát triển kinh tế Việt Nam trong kỉ nguyên AI", "Phiên bản cải tiến: mỗi bài đều có bảng dữ liệu có thể chỉnh, slider tham số, biểu đồ Plotly tương tác, KPI và diễn giải chính sách.", ["12 bài", "LP/MIP", "TOPSIS", "Pareto", "DP", "Stochastic", "Q-learning"])
    c1,c2,c3,c4=st.columns(4)
    with c1: kpi("Dữ liệu", "2020–2025", "Macro · sector · region")
    with c2: kpi("Bài tập", "12", "4 cấp độ")
    with c3: kpi("Biểu đồ", "Plotly", "Zoom · hover · filter")
    with c4: kpi("Thiết kế", "Premium", "Sidebar + cards + tabs")
    st.markdown("### Bản đồ chức năng")
    roadmap=pd.DataFrame({
        "Bài":[1,2,3,4,5,6,7,8,9,10,11,12],
        "Module":["Cobb-Douglas","Linear Programming","Priority Index","Region LP","Project MIP","TOPSIS","Pareto","Dynamic Programming","Labor LP","Stochastic Programming","Q-learning","Integrated AIDEOM"],
        "Tương tác chính":["hệ số co giãn, kịch bản 2030","ngân sách, mức tối thiểu, shadow price","trọng số chính sách, chỉnh dữ liệu ngành","ràng buộc công bằng vùng, heatmap phân bổ","ngân sách, ràng buộc dự án, chọn bắt buộc","trọng số TOPSIS, entropy weights","mục tiêu cạnh tranh, chọn nghiệm thỏa hiệp","chiết khấu, ngân sách theo năm, policy heatmap","tốc độ AI adoption, đào tạo lại","xác suất kịch bản, EVPI/VSS","episode, alpha, gamma, epsilon","5 kịch bản chính sách tích hợp"]
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

# -----------------------------
# BAI 1
# -----------------------------
elif page == "Bài 1 · Cobb-Douglas":
    hero("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng", "Tính TFP, kiểm định dự báo, phân rã tăng trưởng và mô phỏng GDP 2030.", ["TFP", "Growth accounting", "Scenario 2030"])
    df=macro_data().copy()
    with st.sidebar:
        st.subheader("Tham số co giãn")
        alpha=st.slider("α · vốn K",0.05,0.60,0.33,0.01)
        beta=st.slider("β · lao động L",0.05,0.70,0.42,0.01)
        gamma=st.slider("γ · số hóa D",0.00,0.30,0.10,0.01)
        delta=st.slider("δ · AI",0.00,0.25,0.08,0.01)
        theta=st.slider("θ · nhân lực H",0.00,0.25,0.07,0.01)
        normalize=st.checkbox("Chuẩn hóa để tổng hệ số = 1", True)
        st.subheader("Kịch bản 2030")
        D2030=st.slider("D 2030 (% GDP)",20.0,40.0,30.0,0.5)
        AI2030=st.slider("AI 2030 (nghìn DN)",80.0,160.0,100.0,1.0)
        H2030=st.slider("H 2030 (% lao động đào tạo)",28.0,50.0,35.0,0.5)
        gK=st.slider("Tăng K/năm",0.0,0.12,0.06,0.005)
        gL=st.slider("Tăng L/năm",-0.02,0.04,0.006,0.001)
        gA=st.slider("Tăng TFP/năm",0.0,0.04,0.012,0.001)
    coeff=np.array([alpha,beta,gamma,delta,theta],float)
    if normalize: coeff=coeff/coeff.sum()
    alpha,beta,gamma,delta,theta=coeff
    Y=df.GDP_trillion_VND.values; K=df.K_capital.values; L=df.L_labor_million.values; D=df.D_digital_pct.values; AI=df.AI_firms_thousand.values; H=df.H_trained_labor_pct.values
    A=Y/(K**alpha*L**beta*D**gamma*AI**delta*H**theta)
    Abar=A.mean(); Yhat=Abar*(K**alpha*L**beta*D**gamma*AI**delta*H**theta); mape=np.mean(np.abs((Y-Yhat)/Y))*100
    K30=K[-1]*(1+gK)**5; L30=L[-1]*(1+gL)**5; A30=A[-1]*(1+gA)**5
    Y30=A30*(K30**alpha)*(L30**beta)*(D2030**gamma)*(AI2030**delta)*(H2030**theta)
    dln=lambda x: np.log(x[-1]/x[0])/5
    comps={"TFP":dln(A),"K":alpha*dln(K),"L":beta*dln(L),"D":gamma*dln(D),"AI":delta*dln(AI),"H":theta*dln(H)}
    total=sum(comps.values())
    k1,k2,k3,k4=st.columns(4)
    with k1: kpi("TFP 2025", f"{A[-1]:.2f}", f"2020: {A[0]:.2f}")
    with k2: kpi("MAPE", f"{mape:.2f}%", "dự báo với A trung bình")
    with k3: kpi("GDP 2030", f"{Y30:,.0f}", "nghìn tỷ VND")
    with k4: kpi("Tổng hệ số", f"{coeff.sum():.2f}", "lợi suất theo quy mô")
    tab1,tab2,tab3,tab4=st.tabs(["📈 TFP & dự báo", "🧮 Phân rã tăng trưởng", "🗂️ Bảng dữ liệu", "📝 Chính sách"])
    with tab1:
        plot=pd.DataFrame({"Năm":df.year,"TFP":A,"GDP thực tế":Y,"GDP dự báo":Yhat})
        fig=px.line(plot,x="Năm",y=["TFP"],markers=True,title="Xu hướng TFP A_t")
        st.plotly_chart(clean_fig(fig),use_container_width=True)
        fig2=px.line(plot,x="Năm",y=["GDP thực tế","GDP dự báo"],markers=True,title="So sánh GDP thực tế và dự báo")
        st.plotly_chart(clean_fig(fig2),use_container_width=True)
    with tab2:
        cdf=pd.DataFrame({"Yếu tố":list(comps.keys()),"Đóng góp log/năm":list(comps.values()),"Tỷ trọng %":[v/total*100 for v in comps.values()]})
        st.dataframe(cdf,use_container_width=True,hide_index=True)
        fig=px.bar(cdf,x="Yếu tố",y="Tỷ trọng %",text="Tỷ trọng %",title="Tỷ trọng đóng góp vào tăng trưởng GDP 2020–2025")
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(clean_fig(fig),use_container_width=True)
    with tab3:
        st.data_editor(df,use_container_width=True,hide_index=True)
    with tab4:
        st.markdown(f"<div class='policy'>TFP chiếm khoảng <b>{comps['TFP']/total*100:.1f}%</b> tăng trưởng log bình quân. Nếu mục tiêu kinh tế số 30% GDP đạt được, mô hình dự báo GDP 2030 khoảng <b>{Y30:,.0f}</b> nghìn tỷ VND. Hàm ý: chính sách không chỉ tăng vốn K mà phải nâng chất lượng TFP, nhân lực số và năng lực hấp thụ AI.</div>", unsafe_allow_html=True)

# -----------------------------
# BAI 2
# -----------------------------
elif page == "Bài 2 · LP ngân sách":
    hero("Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách", "Giải LP 4 hạng mục đầu tư số, phân tích độ nhạy ngân sách và shadow price.", ["LP", "linprog", "sensitivity"])
    with st.sidebar:
        B=st.slider("Ngân sách tổng B",70,160,100,5)
        minI=st.slider("Sàn hạ tầng số",0,50,25,1)
        minAI=st.slider("Sàn AI & dữ liệu",0,50,15,1)
        minH=st.slider("Sàn nhân lực số",0,50,20,1)
        minR=st.slider("Sàn R&D",0,50,10,1)
        strategic=st.slider("Tỷ trọng AI+R&D tối thiểu",0.0,0.7,0.35,0.05)
        coef=st.data_editor(pd.DataFrame({"Hạng mục":["Hạ tầng","AI & dữ liệu","Nhân lực","R&D"],"Hệ số GDP":[0.85,1.20,0.95,1.35]}),hide_index=True,use_container_width=True)
    c=-coef["Hệ số GDP"].values
    A_ub=[[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[strategic,-(1-strategic),strategic,-(1-strategic)]]
    b_ub=[B,-minI,-minAI,-minH,-minR,0]
    res=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None)]*4,method="highs")
    if not res.success:
        st.error("Bài toán không khả thi với bộ ràng buộc hiện tại.")
    else:
        x=res.x; z=-res.fun
        k1,k2,k3=st.columns(3)
        with k1: kpi("Z*", f"{z:,.1f}", "nghìn tỷ GDP kỳ vọng")
        with k2: kpi("AI+R&D", f"{(x[1]+x[3])/x.sum()*100:.1f}%", "tỷ trọng chiến lược")
        with k3: kpi("Shadow price xấp xỉ", f"{max(coef['Hệ số GDP']):.2f}", "khi ngân sách tăng biên")
        alloc=pd.DataFrame({"Hạng mục":coef["Hạng mục"],"Phân bổ tối ưu":x,"Mức tối thiểu":[minI,minAI,minH,minR],"Hệ số":coef["Hệ số GDP"]})
        c1,c2=st.columns([1,1])
        with c1: st.dataframe(alloc,use_container_width=True,hide_index=True)
        with c2:
            fig=px.bar(alloc,x="Hạng mục",y=["Phân bổ tối ưu","Mức tối thiểu"],barmode="group",title="Phân bổ tối ưu so với sàn bắt buộc")
            st.plotly_chart(clean_fig(fig),use_container_width=True)
        rows=[]
        for BB in range(70,181,10):
            rr=linprog(c,A_ub=A_ub,b_ub=[BB,-minI,-minAI,-minH,-minR,0],bounds=[(0,None)]*4,method="highs")
            rows.append({"B":BB,"Z*":(-rr.fun if rr.success else np.nan)})
        sens=pd.DataFrame(rows)
        fig=px.line(sens,x="B",y="Z*",markers=True,title="Đường cong giá trị tối ưu Z*(B)")
        st.plotly_chart(clean_fig(fig),use_container_width=True)
        st.markdown("<div class='policy'>Khi hệ số R&D cao nhất, nghiệm LP thường dồn phần ngân sách dư vào R&D sau khi thỏa các sàn tối thiểu. Đây là kết quả hợp lý về mặt toán học nhưng cần kiểm tra năng lực hấp thụ, tiến độ giải ngân và ràng buộc nhân lực trong thực tế.</div>", unsafe_allow_html=True)

# -----------------------------
# BAI 3
# -----------------------------
elif page == "Bài 3 · Priority ngành":
    hero("Bài 3 — Chỉ số ưu tiên ngành Priority", "Chuẩn hóa min-max, điều chỉnh trọng số chính sách và xếp hạng 10 ngành.", ["MCDM", "Min-max", "Sensitivity"])
    ds=sectors_data()
    with st.sidebar:
        st.subheader("Trọng số chính sách")
        labels=["growth","productivity","spillover","export","employment","ai_readiness","risk"]
        default=[.15,.15,.20,.15,.10,.20,.15]
        weights=np.array([st.slider(l,0.0,0.6,float(d),0.01) for l,d in zip(labels,default)])
        weights=weights/weights.sum()
        st.caption("Trọng số tự chuẩn hóa về tổng 1")
    edit=st.data_editor(ds,use_container_width=True,hide_index=True)
    X=pd.DataFrame({
        "growth":norm_good(edit.growth),"productivity":norm_good(edit.productivity),"spillover":norm_good(edit.spillover),"export":norm_good(edit.export),"employment":norm_good(edit.employment),"ai_readiness":norm_good(edit.ai_readiness),"risk_good":norm_bad(edit.automation_risk)})
    score=X.values@weights
    rank=edit[["sector"]].copy(); rank["Priority"]=score; rank=rank.sort_values("Priority",ascending=False).reset_index(drop=True); rank["Rank"]=rank.index+1
    top=rank.head(3)
    c1,c2,c3=st.columns(3)
    with c1: kpi("Top 1", top.iloc[0].sector, f"score {top.iloc[0].Priority:.3f}")
    with c2: kpi("Top 2", top.iloc[1].sector, f"score {top.iloc[1].Priority:.3f}")
    with c3: kpi("Top 3", top.iloc[2].sector, f"score {top.iloc[2].Priority:.3f}")
    tab1,tab2,tab3=st.tabs(["🏆 Xếp hạng", "🔥 Heatmap chuẩn hóa", "🧪 Độ nhạy AI"])
    with tab1:
        st.dataframe(rank[["Rank","sector","Priority"]],use_container_width=True,hide_index=True)
        fig=px.bar(rank.sort_values("Priority"),x="Priority",y="sector",orientation="h",title="Xếp hạng ưu tiên chuyển đổi số và AI")
        st.plotly_chart(clean_fig(fig,520),use_container_width=True)
    with tab2:
        heat=X.copy(); heat.index=edit.sector
        fig=px.imshow(heat,aspect="auto",title="Ma trận chỉ tiêu đã chuẩn hóa")
        st.plotly_chart(clean_fig(fig,580),use_container_width=True)
    with tab3:
        rows=[]
        for wai in np.arange(.05,.401,.05):
            w=weights.copy(); w[5]=wai; w=w/w.sum(); sc=X.values@w; order=edit.assign(score=sc).sort_values("score",ascending=False).sector.head(3).tolist()
            rows.append({"w_AI":wai,"Top 1":order[0],"Top 2":order[1],"Top 3":order[2]})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("<div class='policy'>Chỉ số Priority hữu ích vì minh bạch hóa lựa chọn chính sách: thay đổi trọng số sẽ làm thay đổi ngành ưu tiên. Do đó bộ trọng số nên được quyết định qua hội đồng chính sách và đối thoại công khai, không chỉ bởi kỹ thuật.</div>", unsafe_allow_html=True)

# -----------------------------
# BAI 4
# -----------------------------
elif page == "Bài 4 · LP ngành-vùng":
    hero("Bài 4 — LP phân bổ ngân sách số theo vùng", "Mô hình 6 vùng × 4 hạng mục với ràng buộc công bằng số hóa vùng miền.", ["24 biến", "fairness constraint", "heatmap"])
    regs=regions_data(); codes=regs.code.tolist(); items=["I","D","AI","H"]
    beta=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],[1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
    with st.sidebar:
        B=st.slider("Ngân sách tổng",30000,80000,50000,1000)
        floor=st.slider("Sàn mỗi vùng",0,12000,5000,500)
        cap=st.slider("Trần mỗi vùng",6000,20000,12000,500)
        minH=st.slider("Sàn nhân lực số toàn quốc",0,25000,12000,500)
        lam=st.slider("λ công bằng",0.4,0.95,0.70,0.05)
        gamma=st.slider("γ hiệu quả tăng D",0.0005,0.005,0.002,0.0005,format="%.4f")
    n=24; c=np.r_[-beta.flatten(),0]
    Aub=[]; bub=[]
    Aub.append(np.r_[np.ones(n),0]); bub.append(B)
    for r in range(6):
        row=np.zeros(n+1); row[r*4:(r+1)*4]=-1; Aub.append(row); bub.append(-floor)
        row=np.zeros(n+1); row[r*4:(r+1)*4]=1; Aub.append(row); bub.append(cap)
    row=np.zeros(n+1); row[3::4]=-1; Aub.append(row); bub.append(-minH)
    D0=regs.digital_index.values
    for r in range(6):
        row=np.zeros(n+1); row[r*4+1]=gamma; row[-1]=-1; Aub.append(row); bub.append(-D0[r])
        row=np.zeros(n+1); row[r*4+1]=-gamma; row[-1]=lam; Aub.append(row); bub.append(D0[r])
    res=linprog(c,A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*(n+1),method="highs")
    if not res.success:
        st.error("Không khả thi. Hãy giảm sàn vùng hoặc λ công bằng.")
    else:
        x=res.x[:n].reshape(6,4); Z=-res.fun
        df_alloc=pd.DataFrame(x,index=regs.region,columns=["Hạ tầng I","CĐS DN D","AI","Nhân lực H"])
        k1,k2,k3=st.columns(3)
        with k1: kpi("Z*", f"{Z:,.0f}", "GDP gain kỳ vọng")
        with k2: kpi("Vùng nhận nhiều nhất", df_alloc.sum(axis=1).idxmax(), f"{df_alloc.sum(axis=1).max():,.0f}")
        with k3: kpi("M sau đầu tư", f"{res.x[-1]:.1f}", "digital max")
        c1,c2=st.columns([1.05,.95])
        with c1: st.dataframe(df_alloc.style.format("{:,.0f}"),use_container_width=True)
        with c2:
            fig=px.imshow(df_alloc,aspect="auto",title="Heatmap phân bổ tối ưu")
            st.plotly_chart(clean_fig(fig,520),use_container_width=True)
        after=regs.assign(digital_after=regs.digital_index + gamma*x[:,1])
        fig=px.bar(after,x="region",y=["digital_index","digital_after"],barmode="group",title="Chỉ số số hóa trước/sau đầu tư vào D")
        st.plotly_chart(clean_fig(fig,440),use_container_width=True)
        st.markdown("<div class='policy'>Ràng buộc công bằng làm mô hình chuyển một phần ngân sách về vùng yếu hơn để thu hẹp khoảng cách số. Đây là 'chi phí hiệu quả' cần đánh đổi với ổn định xã hội và phát triển bao trùm.</div>", unsafe_allow_html=True)

# -----------------------------
# BAI 5
# -----------------------------
elif page == "Bài 5 · MIP dự án":
    hero("Bài 5 — MIP lựa chọn dự án chuyển đổi số", "Chọn tập dự án tối ưu với biến nhị phân, ràng buộc loại trừ, tiên quyết và ngân sách đa năm.", ["Binary", "Knapsack", "Dependency"])
    pr=projects_data()
    with st.sidebar:
        B=st.slider("Ngân sách 5 năm",50000,120000,80000,1000)
        B12=st.slider("Ngân sách năm 1–2",25000,70000,40000,1000)
        minN=st.slider("Số dự án tối thiểu",1,15,7,1)
        maxN=st.slider("Số dự án tối đa",1,15,11,1)
        require_p1p2=st.checkbox("Bắt buộc có cả P1 và P2",False)
        risk_adjust=st.checkbox("Tối đa hóa lợi ích kỳ vọng theo xác suất hoàn thành",False)
    p=pr.copy()
    prob=p.field.map(lambda f:0.85 if f=="Hạ tầng" else (0.75 if f=="Chính phủ số" else (0.65 if f in ["AI","Bán dẫn"] else 0.80))).values
    benefit=p.benefit.values*(prob if risk_adjust else 1)
    best=None
    for mask in product([0,1], repeat=15):
        y=np.array(mask)
        if y.sum()<minN or y.sum()>maxN: continue
        if (p.cost.values@y)>B or (p.y12.values@y)>B12: continue
        if not require_p1p2 and y[0]+y[1]>1: continue
        if require_p1p2 and not (y[0]==1 and y[1]==1): continue
        if y[7]>y[11] or y[12]>y[11]: continue
        if y[3]+y[4]<1 or y[13]<1: continue
        z=benefit@y
        if best is None or z>best[0]: best=(z,y)
    if best is None:
        st.error("Không có tập dự án khả thi.")
    else:
        z,y=best; selected=p[y==1].copy(); selected["prob_finish"]=prob[y==1]; selected["benefit_used"]=benefit[y==1]
        k1,k2,k3,k4=st.columns(4)
        with k1: kpi("Z*", f"{z:,.0f}", "NPV hoặc E[NPV]")
        with k2: kpi("Chi phí", f"{selected.cost.sum():,.0f}", f"/ {B:,.0f}")
        with k3: kpi("Năm 1–2", f"{selected.y12.sum():,.0f}", f"/ {B12:,.0f}")
        with k4: kpi("Số dự án", f"{len(selected)}", f"biên NPV/cost {z/selected.cost.sum():.2f}")
        st.dataframe(selected,use_container_width=True,hide_index=True)
        fig=px.bar(selected,x="id",y=["cost","benefit"],barmode="group",hover_data=["name","field"],title="Dự án được chọn: chi phí và lợi ích")
        st.plotly_chart(clean_fig(fig,480),use_container_width=True)
        field=selected.groupby("field",as_index=False).agg(cost=("cost","sum"),benefit=("benefit","sum"))
        fig=px.treemap(field,path=["field"],values="cost",color="benefit",title="Cơ cấu chi phí theo lĩnh vực")
        st.plotly_chart(clean_fig(fig,500),use_container_width=True)

# -----------------------------
# BAI 6
# -----------------------------
elif page == "Bài 6 · TOPSIS vùng":
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng", "Đánh giá mức độ ưu tiên đầu tư AI theo 8 tiêu chí và so sánh trọng số chuyên gia với Entropy.", ["TOPSIS", "Entropy", "MCDM"])
    r=regions_data(); criteria=["grdp_pc","fdi","digital_index","ai_readiness","trained_labor","rd_intensity","internet","gini"]
    with st.sidebar:
        st.subheader("Trọng số chuyên gia")
        defaults=[.10,.10,.15,.20,.15,.15,.05,.10]
        w=np.array([st.slider(c,0.0,0.5,float(d),0.01) for c,d in zip(criteria,defaults)])
        w=w/w.sum()
        gini_cost=st.checkbox("Gini là tiêu chí chi phí",True)
    def topsis(X,w,is_benefit):
        R=X/np.sqrt((X**2).sum(axis=0)); V=R*w
        Apos=np.where(is_benefit,V.max(axis=0),V.min(axis=0)); Aneg=np.where(is_benefit,V.min(axis=0),V.max(axis=0))
        sp=np.sqrt(((V-Apos)**2).sum(axis=1)); sn=np.sqrt(((V-Aneg)**2).sum(axis=1))
        return sn/(sp+sn)
    X=r[criteria].values.astype(float); isben=np.array([True]*7+[not gini_cost])
    score=topsis(X,w,isben); rank=r.assign(TOPSIS=score).sort_values("TOPSIS",ascending=False).reset_index(drop=True); rank["Rank"]=rank.index+1
    # entropy weights with benefit-adjusted minmax
    Xmm=np.column_stack([norm_good(r[c]) if i<7 or not gini_cost else norm_bad(r[c]) for i,c in enumerate(criteria)])+1e-9
    P=Xmm/Xmm.sum(axis=0); E=-(P*np.log(P)).sum(axis=0)/np.log(len(Xmm)); went=(1-E)/(1-E).sum()
    ent_score=topsis(X,went,isben); ent_rank=r.assign(Entropy_TOPSIS=ent_score).sort_values("Entropy_TOPSIS",ascending=False).reset_index(drop=True)
    c1,c2,c3=st.columns(3)
    with c1: kpi("Top chuyên gia", rank.iloc[0].region, f"{rank.iloc[0].TOPSIS:.3f}")
    with c2: kpi("Top entropy", ent_rank.iloc[0].region, f"{ent_rank.iloc[0].Entropy_TOPSIS:.3f}")
    with c3: kpi("w_AI", f"{w[3]:.2f}", "trọng số AI readiness")
    tab1,tab2,tab3=st.tabs(["🏆 Xếp hạng", "⚖️ Trọng số", "🧪 Độ nhạy w_AI"])
    with tab1:
        st.dataframe(rank[["Rank","region","TOPSIS"]],use_container_width=True,hide_index=True)
        fig=px.bar(rank.sort_values("TOPSIS"),x="TOPSIS",y="region",orientation="h",title="TOPSIS score theo vùng")
        st.plotly_chart(clean_fig(fig,500),use_container_width=True)
    with tab2:
        wdf=pd.DataFrame({"criteria":criteria,"Expert":w,"Entropy":went})
        fig=px.bar(wdf,x="criteria",y=["Expert","Entropy"],barmode="group",title="So sánh trọng số chuyên gia và Entropy")
        st.plotly_chart(clean_fig(fig,460),use_container_width=True)
        st.dataframe(wdf,use_container_width=True,hide_index=True)
    with tab3:
        rows=[]
        for wai in np.arange(.10,.401,.05):
            ww=w.copy(); ww[3]=wai; ww=ww/ww.sum(); sc=topsis(X,ww,isben); top3=r.assign(sc=sc).sort_values("sc",ascending=False).region.head(3).tolist()
            rows.append({"w_AI":wai,"Top 1":top3[0],"Top 2":top3[1],"Top 3":top3[2]})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# -----------------------------
# BAI 7
# -----------------------------
elif page == "Bài 7 · Pareto NSGA-II":
    hero("Bài 7 — Tối ưu đa mục tiêu Pareto", "Mô phỏng tập nghiệm Pareto cho 4 mục tiêu cạnh tranh: GDP, bao trùm, môi trường, an ninh dữ liệu.", ["Pareto", "4 objectives", "TOPSIS compromise"])
    with st.sidebar:
        n=st.slider("Số phương án mô phỏng",500,8000,2500,500)
        seed=st.number_input("Seed",0,9999,42)
        wgdp=st.slider("Ưu tiên GDP",0.0,1.0,0.35,0.05)
        winc=st.slider("Ưu tiên bao trùm",0.0,1.0,0.25,0.05)
        wenv=st.slider("Ưu tiên môi trường",0.0,1.0,0.20,0.05)
        wcyb=st.slider("Ưu tiên an ninh dữ liệu",0.0,1.0,0.20,0.05)
    rng=np.random.default_rng(seed); alloc=rng.dirichlet([2,2,1.5,1.2],size=n)*100
    I,D,AI,H=alloc.T
    GDP=0.8*I+1.15*D+1.45*AI+0.95*H-0.003*AI**2
    Inclusion=0.30*I+0.45*D+0.20*AI+1.05*H
    Emission=0.40*I+0.25*D+0.55*AI+0.15*H
    CyberRisk=0.15*I+0.35*D+0.75*AI+0.25*H
    objs=np.column_stack([GDP,Inclusion,-Emission,-CyberRisk])
    dominated=np.zeros(n,dtype=bool)
    # Approx Pareto for n up to 8000: compare with random subset chunks
    for i in range(n):
        if dominated[i]: continue
        better=(objs>=objs[i]).all(axis=1)&(objs>objs[i]).any(axis=1)
        if better.any(): dominated[i]=True
    pareto=pd.DataFrame(alloc[~dominated],columns=["I","D","AI","H"])
    pareto["GDP"]=GDP[~dominated]; pareto["Inclusion"]=Inclusion[~dominated]; pareto["Emission"]=Emission[~dominated]; pareto["CyberRisk"]=CyberRisk[~dominated]
    ww=np.array([wgdp,winc,wenv,wcyb]); ww=ww/ww.sum()
    score=norm_good(pareto.GDP)*ww[0]+norm_good(pareto.Inclusion)*ww[1]+norm_bad(pareto.Emission)*ww[2]+norm_bad(pareto.CyberRisk)*ww[3]
    best=pareto.iloc[int(np.argmax(score))]
    c1,c2,c3=st.columns(3)
    with c1: kpi("Pareto points", f"{len(pareto)}", f"/ {n} phương án")
    with c2: kpi("GDP tốt nhất", f"{pareto.GDP.max():.1f}", "trên biên Pareto")
    with c3: kpi("Nghiệm thỏa hiệp", f"AI {best.AI:.1f}%", "theo trọng số")
    fig=px.scatter_3d(pareto,x="GDP",y="Inclusion",z="Emission",color="CyberRisk",hover_data=["I","D","AI","H"],title="Biên Pareto 3D: GDP - Bao trùm - Phát thải")
    st.plotly_chart(clean_fig(fig,650),use_container_width=True)
    st.dataframe(pareto.sort_values("GDP",ascending=False).head(20),use_container_width=True,hide_index=True)
    st.markdown(f"<div class='policy'>Nghiệm thỏa hiệp gợi ý phân bổ I={best.I:.1f}, D={best.D:.1f}, AI={best.AI:.1f}, H={best.H:.1f}. Điểm quan trọng: nghiệm 'tốt nhất' phụ thuộc vào trọng số xã hội, vì các mục tiêu xung đột nhau.</div>", unsafe_allow_html=True)

# -----------------------------
# BAI 8
# -----------------------------
elif page == "Bài 8 · Dynamic Programming":
    hero("Bài 8 — Tối ưu liên thời gian 2026–2035", "Dynamic programming chọn đầu tư số theo thời gian để tối đa hóa phúc lợi chiết khấu.", ["DP", "policy heatmap", "discounting"])
    with st.sidebar:
        horizon=st.slider("Số năm",5,10,10,1)
        annual=st.slider("Ngân sách/năm",20,100,50,5)
        rho=st.slider("Hệ số chiết khấu β",0.80,0.99,0.94,0.01)
        max_state=st.slider("Digital index max",60,120,100,5)
    states=np.arange(30,max_state+1,5); actions=np.arange(0,annual+1,10)
    V=np.zeros((horizon+1,len(states))); policy=np.zeros((horizon,len(states)))
    for t in reversed(range(horizon)):
        for si,s in enumerate(states):
            vals=[]
            for a in actions:
                ns=min(max_state, s+0.18*a+1.0)
                sj=int(np.argmin(abs(states-ns)))
                reward=0.9*s+1.4*a-0.010*a*a-0.03*max(0,70-s)**2
                vals.append(reward+rho*V[t+1,sj])
            bi=int(np.argmax(vals)); V[t,si]=vals[bi]; policy[t,si]=actions[bi]
    k1,k2,k3=st.columns(3)
    with k1: kpi("V0", f"{V[0,0]:.1f}", "phúc lợi chiết khấu")
    with k2: kpi("Action đầu", f"{policy[0,0]:.0f}", "nghìn tỷ")
    with k3: kpi("β", f"{rho:.2f}", "ưu tiên hiện tại/tương lai")
    polydf=pd.DataFrame(policy,columns=states,index=[2026+i for i in range(horizon)])
    fig=px.imshow(polydf,aspect="auto",labels=dict(x="Digital index state",y="Năm",color="Đầu tư"),title="Policy heatmap: đầu tư tối ưu theo trạng thái và năm")
    st.plotly_chart(clean_fig(fig,560),use_container_width=True)
    traj=[]; s=38
    for t in range(horizon):
        si=int(np.argmin(abs(states-s))); a=policy[t,si]; traj.append({"year":2026+t,"digital_index":s,"investment":a}); s=min(max_state,s+0.18*a+1)
    tr=pd.DataFrame(traj)
    fig=px.line(tr,x="year",y=["digital_index","investment"],markers=True,title="Quỹ đạo mô phỏng từ trạng thái D=38")
    st.plotly_chart(clean_fig(fig,460),use_container_width=True)
    st.dataframe(tr,use_container_width=True,hide_index=True)

# -----------------------------
# BAI 9
# -----------------------------
elif page == "Bài 9 · AI & lao động":
    hero("Bài 9 — Tác động AI tới lao động ngành", "Mô phỏng thay thế việc làm, tạo việc mới và nhu cầu đào tạo lại theo ngành.", ["Labor", "AI adoption", "reskilling"])
    base=sectors_data().iloc[[0,1,4,5,6,7,8,9]].copy()
    with st.sidebar:
        adoption=st.slider("Tốc độ ứng dụng AI",0.1,1.5,0.8,0.05)
        train_eff=st.slider("Hiệu quả đào tạo lại",0.2,1.0,0.6,0.05)
        newjob_factor=st.slider("Hệ số tạo việc mới",0.05,0.60,0.25,0.05)
    displaced=base.employment*base.automation_risk/100*adoption*0.35
    newjobs=base.employment*(base.ai_readiness/100)*newjob_factor
    reskill_need=np.maximum(displaced-newjobs,0)/train_eff
    out=pd.DataFrame({"Ngành":base.sector,"Việc làm hiện tại":base.employment,"Bị thay thế":displaced,"Việc mới":newjobs,"Net job":newjobs-displaced,"Cần đào tạo lại":reskill_need})
    k1,k2,k3=st.columns(3)
    with k1: kpi("Bị thay thế", f"{out['Bị thay thế'].sum():.2f}tr", "triệu lao động")
    with k2: kpi("Việc mới", f"{out['Việc mới'].sum():.2f}tr", "triệu lao động")
    with k3: kpi("Net job", f"{out['Net job'].sum():+.2f}tr", "việc mới - thay thế")
    st.dataframe(out,use_container_width=True,hide_index=True)
    fig=px.bar(out,x="Ngành",y=["Bị thay thế","Việc mới","Net job"],barmode="group",title="Tác động AI tới lao động theo ngành")
    st.plotly_chart(clean_fig(fig,560),use_container_width=True)
    fig=px.bar(out.sort_values("Cần đào tạo lại",ascending=False),x="Ngành",y="Cần đào tạo lại",title="Nhu cầu đào tạo lại theo ngành")
    st.plotly_chart(clean_fig(fig,430),use_container_width=True)

# -----------------------------
# BAI 10
# -----------------------------
elif page == "Bài 10 · Stochastic LP":
    hero("Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn", "So sánh quyết định dưới bất định: SP, EV, WS, VSS và EVPI.", ["Stochastic", "Scenario", "EVPI/VSS"])
    with st.sidebar:
        p_low=st.slider("Xác suất kịch bản thấp",0.0,1.0,0.25,0.05)
        p_mid=st.slider("Xác suất kịch bản cơ sở",0.0,1.0,0.50,0.05)
        p_high=max(0,1-p_low-p_mid); st.caption(f"Xác suất cao tự tính: {p_high:.2f}")
        B=st.slider("Ngân sách giai đoạn 1",50,150,100,5)
    probs=np.array([p_low,p_mid,p_high]); probs=probs/probs.sum() if probs.sum()>0 else np.array([.25,.5,.25])
    scen=pd.DataFrame({"scenario":["Thấp","Cơ sở","Cao"],"prob":probs,"mult_D":[0.75,1.0,1.25],"mult_AI":[0.65,1.0,1.40],"shock":[-10,0,12]})
    grid=np.arange(0,B+1,10); best=(-1,None); ws=0
    for I in grid:
      for D in grid:
       for AIv in grid:
        H=B-I-D-AIv
        if H<0: continue
        vals=[]
        for _,s in scen.iterrows():
            val=0.8*I+1.1*D*s.mult_D+1.35*AIv*s.mult_AI+0.9*H+s.shock-0.004*max(0,AIv-35)**2
            vals.append(val)
        ev=np.dot(probs,vals)
        if ev>best[0]: best=(ev,(I,D,AIv,H,vals))
    for _,s in scen.iterrows():
        b=(-1,None)
        for I in grid:
          for D in grid:
           for AIv in grid:
            H=B-I-D-AIv
            if H<0: continue
            val=0.8*I+1.1*D*s.mult_D+1.35*AIv*s.mult_AI+0.9*H+s.shock-0.004*max(0,AIv-35)**2
            if val>b[0]: b=(val,(I,D,AIv,H))
        ws+=s.prob*b[0]
    sp=best[0]; evpi=ws-sp
    k1,k2,k3=st.columns(3)
    with k1: kpi("SP value", f"{sp:.1f}", "expected value")
    with k2: kpi("WS value", f"{ws:.1f}", "perfect information")
    with k3: kpi("EVPI", f"{evpi:.1f}", "giá trị thông tin hoàn hảo")
    alloc=pd.DataFrame({"Hạng mục":["I","D","AI","H"],"Giai đoạn 1 tối ưu":best[1][:4]})
    st.dataframe(scen,use_container_width=True,hide_index=True)
    fig=px.pie(alloc,names="Hạng mục",values="Giai đoạn 1 tối ưu",title="Phân bổ giai đoạn 1 tối ưu dưới bất định")
    st.plotly_chart(clean_fig(fig,480),use_container_width=True)

# -----------------------------
# BAI 11
# -----------------------------
elif page == "Bài 11 · Q-learning":
    hero("Bài 11 — Học tăng cường Tabular Q-learning", "Huấn luyện tác nhân chọn chính sách chuyển đổi số qua 81 trạng thái và 5 hành động.", ["RL", "Q-table", "learning curve"])
    with st.sidebar:
        episodes=st.slider("Số episode",500,8000,3000,500)
        alpha=st.slider("α learning rate",0.01,0.5,0.10,0.01)
        gamma=st.slider("γ discount",0.70,0.99,0.95,0.01)
        eps0=st.slider("ε ban đầu",0.05,0.8,0.30,0.05)
        seed=st.number_input("Seed",0,9999,42)
    rng=np.random.default_rng(seed); nS=81; nA=5; Q=np.zeros((nS,nA)); rewards=[]
    def step(s,a):
        g=s//27; d=(s//9)%3; u=(s//3)%3; risk=s%3
        if a==0: g2=max(0,g-1); d2=d; u2=u; r2=risk
        elif a==1: g2=g; d2=min(2,d+1); u2=max(0,u-1); r2=risk
        elif a==2: g2=min(2,g+1); d2=min(2,d+1); u2=u; r2=min(2,risk+1)
        elif a==3: g2=g; d2=d; u2=max(0,u-1); r2=max(0,risk-1)
        else: g2=g; d2=d; u2=u; r2=max(0,risk-1)
        ns=g2*27+d2*9+u2*3+r2
        reward=4*g2+3*d2-2*u2-2.5*r2-[0,1,2.5,1.5,2][a]+rng.normal(0,.4)
        return ns,reward
    for ep in range(episodes):
        s=rng.integers(0,nS); total=0; eps=eps0*(0.995**ep)
        for t in range(30):
            a=rng.integers(0,nA) if rng.random()<eps else int(Q[s].argmax())
            ns,r=step(s,a); Q[s,a]+=alpha*(r+gamma*Q[ns].max()-Q[s,a]); s=ns; total+=r
        rewards.append(total)
    rw=pd.DataFrame({"episode":np.arange(episodes),"reward":rewards}); rw["smooth"]=rw.reward.rolling(100,min_periods=1).mean()
    actions=["truyền thống","cân bằng","số hóa nhanh","AI dẫn dắt","bao trùm"]
    k1,k2,k3=st.columns(3)
    with k1: kpi("Mean reward cuối", f"{np.mean(rewards[-100:]):.2f}", "100 episode")
    with k2: kpi("Best action phổ biến", actions[np.bincount(Q.argmax(axis=1),minlength=5).argmax()], "trong Q-table")
    with k3: kpi("Q states", "81", "3×3×3×3")
    fig=px.line(rw,x="episode",y=["reward","smooth"],title="Learning curve")
    st.plotly_chart(clean_fig(fig,520),use_container_width=True)
    pol=pd.DataFrame({"state":np.arange(nS),"best_action":[actions[i] for i in Q.argmax(axis=1)],"max_Q":Q.max(axis=1)})
    st.dataframe(pol.head(81),use_container_width=True,hide_index=True)

# -----------------------------
# BAI 12
# -----------------------------
elif page == "Bài 12 · Dashboard tích hợp":
    hero("Bài 12 — AIDEOM-VN Dashboard tích hợp", "Kết nối M1–M6: dự báo kinh tế, sẵn sàng số, phân bổ, lao động, rủi ro và 5 kịch bản chính sách.", ["Integrated", "5 scenarios", "M1–M6"])
    scenarios=pd.DataFrame({
        "Kịch bản":["Cơ sở","AI dẫn dắt","Bao trùm","Xanh & an toàn","Tăng tốc số hóa"],
        "D_2030":[25,32,28,27,35],"AI_2030":[95,140,110,115,130],"H_2030":[33,36,42,38,35],"Risk":[0.45,0.62,0.32,0.25,0.55],"Budget":[100,135,120,125,140]
    })
    edit=st.data_editor(scenarios,use_container_width=True,hide_index=True)
    # Composite score
    m=macro_data(); A=m.GDP_trillion_VND.iloc[-1]/(m.K_capital.iloc[-1]**.33*m.L_labor_million.iloc[-1]**.42*m.D_digital_pct.iloc[-1]**.1*m.AI_firms_thousand.iloc[-1]**.08*m.H_trained_labor_pct.iloc[-1]**.07)
    edit["GDP_2030_index"]=100*(edit.D_2030/19.5)**.10*(edit.AI_2030/80.1)**.08*(edit.H_2030/29.2)**.07*(edit.Budget/100)**.20
    edit["Inclusive_index"]=60+edit.H_2030*1.2-edit.Risk*25
    edit["Risk_score"]=100*(1-edit.Risk)
    edit["Composite"]=0.45*norm_good(edit.GDP_2030_index)+0.30*norm_good(edit.Inclusive_index)+0.25*norm_good(edit.Risk_score)
    rank=edit.sort_values("Composite",ascending=False).reset_index(drop=True); rank["Rank"]=rank.index+1
    c1,c2,c3=st.columns(3)
    with c1: kpi("Kịch bản tốt nhất", rank.iloc[0]["Kịch bản"], f"Composite {rank.iloc[0].Composite:.3f}")
    with c2: kpi("GDP index cao nhất", f"{edit.GDP_2030_index.max():.1f}", "so với baseline")
    with c3: kpi("Rủi ro thấp nhất", edit.loc[edit.Risk.idxmin(),"Kịch bản"], f"Risk {edit.Risk.min():.2f}")
    tab1,tab2,tab3,tab4=st.tabs(["📊 Tổng quan", "🧩 M1–M6", "🕸️ Radar", "📝 Kết luận"])
    with tab1:
        st.dataframe(rank[["Rank","Kịch bản","GDP_2030_index","Inclusive_index","Risk_score","Composite"]],use_container_width=True,hide_index=True)
        fig=px.bar(rank,x="Kịch bản",y=["GDP_2030_index","Inclusive_index","Risk_score"],barmode="group",title="So sánh 5 kịch bản chính sách")
        st.plotly_chart(clean_fig(fig,520),use_container_width=True)
    with tab2:
        modules=pd.DataFrame({"Module":["M1","M2","M3","M4","M5","M6"],"Tên":["Dự báo kinh tế","Sẵn sàng số","Tối ưu phân bổ","Lao động","Rủi ro","Dashboard quyết định"],"Đầu vào":["Macro 2020–2025","Sector/Region","Budget + beta","AI adoption","Risk params","5 scenarios"],"Đầu ra":["GDP/TFP 2030","Digital+AI index","Ngân sách ngành-vùng","Net job","Cyber/env/dependency","Policy ranking"],"Kỹ thuật":["Cobb-Douglas","TOPSIS","LP + DP","Labor balance","Pareto + SP","Composite MCDM"]})
        st.dataframe(modules,use_container_width=True,hide_index=True)
    with tab3:
        chosen=st.selectbox("Chọn kịch bản radar", edit["Kịch bản"].tolist(), index=0)
        row=edit[edit["Kịch bản"]==chosen].iloc[0]
        cats=["GDP","Bao trùm","An toàn","Số hóa","AI","Nhân lực"]
        vals=[norm_good(edit.GDP_2030_index)[row.name],norm_good(edit.Inclusive_index)[row.name],norm_good(edit.Risk_score)[row.name],row.D_2030/35,row.AI_2030/140,row.H_2030/42]
        fig=go.Figure(data=go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill='toself',name=chosen))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),title=f"Radar chính sách: {chosen}")
        st.plotly_chart(clean_fig(fig,540),use_container_width=True)
    with tab4:
        st.markdown(f"<div class='policy'>Theo trọng số tích hợp hiện tại, kịch bản <b>{rank.iloc[0]['Kịch bản']}</b> đứng đầu. Khi trình bày bài, nên nhấn mạnh đây không phải 'đáp án tuyệt đối' mà là công cụ ra quyết định: thay trọng số, giả định rủi ro hoặc ngân sách sẽ tạo ra thứ tự ưu tiên khác.</div>", unsafe_allow_html=True)
