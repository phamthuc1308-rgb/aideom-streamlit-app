"""AIDEOM-VN Streamlit application.

A complete interactive dashboard for the 12 exercises in the course
"Mô hình ra quyết định – Phát triển kinh tế Việt Nam trong kỉ nguyên AI".
The app is designed to run locally, on GitHub, and on Streamlit Cloud.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils import bai1_cobb_douglas as b1
from utils import bai2_lp as b2
from utils import bai3_priority as b3
from utils import bai4_lp_vung as b4
from utils import bai5_mip as b5
from utils import bai6_topsis as b6
from utils import bai7_pareto as b7
from utils import bai8_dynamic as b8
from utils import bai9_labor as b9
from utils import bai10_stochastic as b10
from utils import bai11_qlearning as b11
from utils import bai12_integrated as b12
from utils.data_loader import load_macro, load_sectors, load_regions

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide", initial_sidebar_state="expanded")


def load_css() -> None:
    css = APP_DIR / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = "".join([f'<span class="chip">{c}</span>' for c in (chips or [])])
    st.markdown(
        f"""
        <div class="aideom-hero">
          {chips_html}
          <div class="aideom-title">{title}</div>
          <div class="aideom-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analysis(text: str) -> None:
    st.markdown(f'<div class="analysis-box">🔍 <b>Tác nhân phân tích kết quả</b><br>{text}</div>', unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)


def sidebar() -> str:
    st.sidebar.markdown("## 🇻🇳 AIDEOM-VN")
    st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")
    pages = [
        "🏠 Trang chủ",
        "1️⃣ Bài 1 — Cobb-Douglas + AI",
        "2️⃣ Bài 2 — LP ngân sách số",
        "3️⃣ Bài 3 — Priority 10 ngành",
        "4️⃣ Bài 4 — LP ngành-vùng",
        "5️⃣ Bài 5 — MIP 15 dự án",
        "6️⃣ Bài 6 — TOPSIS 6 vùng",
        "7️⃣ Bài 7 — NSGA-II Pareto",
        "8️⃣ Bài 8 — Động 2026-2035",
        "9️⃣ Bài 9 — Lao động & AI",
        "🔟 Bài 10 — Stochastic SP",
        "🤖 Bài 11 — Q-learning RL",
        "🇻🇳 Bài 12 — AIDEOM tích hợp",
    ]
    page = st.sidebar.radio("Chọn bài", pages, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("### Thông tin dữ liệu")
    st.sidebar.write("📁 NSO/GSO, MoST, MIC, MPI, WB, GII 2025")
    st.sidebar.write("🧰 Streamlit · Plotly · SciPy · PuLP · CVXPY · pymoo")
    st.sidebar.write("☁️ Sẵn sàng deploy Streamlit Cloud")
    return page


def page_home():
    hero("AIDEOM-VN Dashboard", "Bộ bài tập Mô hình ra quyết định – Phát triển kinh tế Việt Nam trong kỉ nguyên AI", ["12 bài", "Python", "Tối ưu hóa", "Dashboard"])
    st.markdown("Ứng dụng gồm 12 bài từ Dễ đến Rất khó, có dữ liệu Việt Nam 2020–2025, mô hình toán, biểu đồ Plotly và diễn giải chính sách.")
    c1,c2,c3 = st.columns(3)
    c1.dataframe(load_macro().head(), use_container_width=True)
    c2.dataframe(load_sectors()[["sector_name_vi","growth_rate_2024_pct","ai_readiness_0_100"]], use_container_width=True)
    c3.dataframe(load_regions()[["region_name_vi","digital_index_0_100","ai_readiness_0_100"]], use_container_width=True)
    analysis("Hãy chọn từng bài ở sidebar. Các bài có nút Chạy để tránh tự chạy thuật toán nặng khi thay đổi tham số.")


def page1():
    hero("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng", "Ước lượng TFP, dự báo GDP, phân rã tăng trưởng và kịch bản 2030", ["Cấp độ dễ", "Macro 2020–2025", "TFP"])
    st.latex(r"Y_t=A_tK_t^\alpha L_t^\beta D_t^\gamma AI_t^\delta H_t^\theta")
    st.caption("Dùng slider để điều chỉnh hệ số. App tự chuẩn hóa để tổng hệ số bằng 1.")
    cols = st.columns(5)
    alpha = cols[0].slider("α K",0.05,0.70,0.33,0.01)
    beta = cols[1].slider("β L",0.05,0.70,0.42,0.01)
    gamma = cols[2].slider("γ D",0.01,0.30,0.10,0.01)
    delta = cols[3].slider("δ AI",0.01,0.30,0.08,0.01)
    theta = cols[4].slider("θ H",0.01,0.30,0.07,0.01)
    if st.button("🚀 Chạy Bài 1"):
        df, contrib, mape, y2030, c = b1.compute_model(alpha,beta,gamma,delta,theta)
        metric_row([("MAPE", f"{mape:.2f}%", None), ("GDP 2030 dự báo", f"{y2030:,.1f} nghìn tỷ VND", None), ("Tổng hệ số", f"{sum(c.values()):.2f}", None)])
        st.plotly_chart(b1.fig_tfp(df), use_container_width=True)
        st.plotly_chart(b1.fig_actual_vs_pred(df), use_container_width=True)
        st.plotly_chart(b1.fig_contrib(contrib), use_container_width=True)
        st.dataframe(df[["year","GDP_trillion_VND","TFP_A","Y_hat","APE_pct"]], use_container_width=True)
        st.dataframe(contrib, use_container_width=True)
        analysis(f"TFP trung bình giúp mô hình khớp dữ liệu với MAPE khoảng {mape:.2f}%. Kịch bản 2030 cho thấy GDP nhạy với D, AI và H, nhưng vẫn phụ thuộc nền K và L.")


def page2():
    hero("Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số", "LP 4 biến, giải bằng SciPy và PuLP, phân tích shadow price", ["LP", "SciPy", "PuLP"])
    cols=st.columns(6)
    budget=cols[0].number_input("Ngân sách",50,200,100,10)
    floors=[cols[i+1].number_input(f"Sàn x{i+1}",0,80,v,5) for i,v in enumerate([25,15,20,10])]
    strategic=st.slider("Tỷ trọng AI + R&D tối thiểu",0.20,0.60,0.35,0.01)
    if st.button("🚀 Chạy LP"):
        res, df, z=b2.solve_scipy(budget, tuple(floors), strategic)
        _, shadow=b2.solve_pulp(budget, tuple(floors), strategic)
        metric_row([("Trạng thái", "Khả thi" if res.success else "Không khả thi", None), ("Z*", f"{z:,.2f}", None), ("Tổng phân bổ", f"{df['Phân bổ'].sum():,.2f}", None)])
        st.plotly_chart(b2.fig_allocation(df), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        if not shadow.empty: st.dataframe(shadow, use_container_width=True)
        sens=b2.sensitivity_curve(tuple(floors)); st.plotly_chart(b2.fig_sensitivity(sens), use_container_width=True)
        res30, _, z30 = b2.solve_scipy(budget, (floors[0], floors[1], 30, floors[3]), strategic)
        st.info(f"Trường hợp x3 ≥ 30: {'khả thi' if res30.success else 'không khả thi'}, Z* = {z30:.2f}")
        analysis("Nếu shadow price của ngân sách dương, mỗi đơn vị ngân sách tăng thêm còn tạo thêm GDP gain trong vùng nghiệm hiện tại. R&D thường được ưu tiên do hệ số mục tiêu cao nhất.")


def page3():
    hero("Bài 3 — Priority Index 10 ngành", "Chuẩn hóa min-max, xếp hạng ngành và phân tích độ nhạy trọng số AI", ["MCDM", "Priority", "Heatmap"])
    defaults=[0.15,0.15,0.20,0.15,0.10,0.20]
    cols=st.columns(7)
    weights=[cols[i].slider(b3.LABELS[i],0.0,0.5,defaults[i],0.01) for i in range(6)]
    w_risk=cols[6].slider("Risk penalty",0.0,0.5,0.15,0.01)
    if st.button("🚀 Tính Priority"):
        rank,norm=b3.compute_priority(weights,w_risk)
        st.plotly_chart(b3.fig_rank(rank), use_container_width=True)
        st.dataframe(rank, use_container_width=True)
        with st.expander("Ma trận chuẩn hóa"):
            st.dataframe(norm, use_container_width=True)
        sens=b3.sensitivity_ai(); st.plotly_chart(b3.fig_heatmap(sens), use_container_width=True)
        comp=b3.compare_weight_sets(); st.dataframe(comp.groupby("Bộ trọng số").head(3), use_container_width=True)
        analysis(f"Top-1 hiện tại là {rank.iloc[0]['sector_name_vi']}. Kết quả thay đổi khi tăng trọng số AI readiness, nên cần công khai lựa chọn trọng số trong quá trình governance.")


def page4():
    hero("Bài 4 — LP phân bổ ngân sách số theo vùng", "24 biến, ràng buộc công bằng vùng miền và chi phí kinh tế của công bằng", ["LP vùng", "PuLP", "Công bằng"])
    c1,c2,c3,c4=st.columns(4)
    lam=c1.slider("λ công bằng",0.10,1.00,0.70,0.01)
    floor=c2.number_input("Sàn vùng",0,10000,5000,500)
    cap=c3.number_input("Trần vùng",6000,20000,12000,500)
    budget=c4.number_input("Tổng ngân sách",10000,80000,50000,1000)
    if st.button("🚀 Chạy LP vùng"):
        m,df,z,status=b4.solve_lp(lam,floor,cap,budget)
        if status != "Optimal":
            best=b4.find_max_feasible_lambda(floor,cap)
            st.warning(f"λ={lam:.2f} không tối ưu/khả thi. λ lớn nhất tìm được khoảng {best:.2f}.")
        else:
            metric_row([("Z*",f"{z:,.0f}",None),("Trạng thái",status,None),("Tổng phân bổ",f"{df['Phân bổ'].sum():,.0f}",None)])
            st.plotly_chart(b4.fig_heatmap(df), use_container_width=True)
            st.plotly_chart(b4.fig_region_totals(df), use_container_width=True)
            st.dataframe(df, use_container_width=True)
            st.dataframe(pd.DataFrame([b4.fairness_cost(lam,floor,cap)]), use_container_width=True)
            analysis("Ràng buộc công bằng làm giảm phần nào GDP gain nhưng tránh tập trung vốn vào vùng sẵn sàng số cao. Đây là chi phí kinh tế của mục tiêu bao trùm.")


def page5():
    hero("Bài 5 — MIP lựa chọn 15 dự án", "Biến nhị phân, ràng buộc loại trừ, tiên quyết và ngân sách đa năm", ["MIP", "Knapsack", "CBC"])
    budget=st.slider("Ngân sách tổng 5 năm",50000,110000,80000,5000)
    budget12=st.slider("Ngân sách năm 1–2",25000,60000,40000,2500)
    if st.button("🚀 Chạy MIP"):
        df,z,status=b5.solve_mip(budget,budget12)
        metric_row([("Trạng thái",status,None),("Tổng NPV",f"{z:,.0f}",None),("Số dự án chọn",str(int(df['Chọn'].sum())),None)])
        st.plotly_chart(b5.fig_roi(df), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        df100,z100,st100=b5.solve_mip(100000, budget12)
        st.info(f"Khi ngân sách 100.000: {int(df100['Chọn'].sum())} dự án, Z*={z100:,.0f}, trạng thái={st100}")
        analysis("MIP cho thấy dự án có ROI cao chưa chắc được chọn nếu bị ràng buộc tiên quyết, loại trừ hoặc trần ngân sách 2 năm đầu.")


def page6():
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng", "So sánh trọng số chuyên gia và Entropy cho ưu tiên đầu tư AI", ["TOPSIS", "Entropy", "MCDM"])
    labels=["GRDP/ng", "FDI", "Digital", "AI", "LĐ ĐT", "R&D", "Internet", "Gini"]
    cols=st.columns(8)
    weights=[cols[i].slider(labels[i],0.0,0.5,float(b6.DEFAULT_W[i]),0.01) for i in range(8)]
    if st.button("🚀 Tính TOPSIS"):
        rank,w=b6.topsis(weights)
        comp,ew=b6.compare_methods(weights)
        st.plotly_chart(b6.fig_scores(comp), use_container_width=True)
        st.dataframe(rank, use_container_width=True)
        st.write("Trọng số Entropy:", pd.DataFrame({"Tiêu chí":labels,"Entropy weight":ew}))
        analysis(f"Vùng dẫn đầu theo trọng số hiện tại là {rank.iloc[0]['region_name_vi']}. Nếu dùng Entropy, tiêu chí có độ phân tán lớn sẽ được tăng trọng số khách quan.")


def page7():
    hero("Bài 7 — Tối ưu đa mục tiêu Pareto", "4 mục tiêu: tăng trưởng, bao trùm, môi trường, an ninh dữ liệu", ["NSGA-II/Pareto", "TOPSIS", "Đa mục tiêu"])
    c=st.columns(5)
    n=c[0].slider("Số nghiệm mô phỏng",100,1500,500,100)
    weights=[c[i+1].slider(w,0.0,0.8,v,0.01) for i,(w,v) in enumerate(zip(["w GDP","w bao trùm","w môi trường","w an ninh"],[0.40,0.25,0.20,0.15]))]
    if st.button("🚀 Chạy Pareto"):
        with st.status("Đang sinh tập nghiệm Pareto...", expanded=False):
            pareto,_=b7.run_pareto(n)
        scored,best,cost=b7.choose_topsis(pareto,weights)
        metric_row([("Số nghiệm Pareto",str(len(pareto)),None),("Nghiệm thỏa hiệp",str(int(best.solution_id)),None),("GDP gain",f"{best.GDP_gain:,.0f}",None)])
        st.plotly_chart(b7.fig_3d(scored), use_container_width=True)
        st.plotly_chart(b7.fig_parallel(scored.head(100)), use_container_width=True)
        st.dataframe(scored.head(20), use_container_width=True)
        st.dataframe(pd.DataFrame([cost]), use_container_width=True)
        analysis("Không có một nghiệm tuyệt đối tốt nhất. TOPSIS giúp chọn nghiệm thỏa hiệp theo ưu tiên chính sách, nhưng quyết định cuối vẫn cần thảo luận chính trị - xã hội.")


def page8():
    hero("Bài 8 — Tối ưu động 2026–2035", "SLSQP cho phân bổ liên thời gian, so sánh đầu tư đều và front-load", ["Dynamic", "SLSQP", "Welfare"])
    c=st.columns(4)
    rho=c[0].slider("ρ chiết khấu",0.80,0.995,0.97,0.005)
    dK=c[1].slider("δK",0.01,0.12,0.05,0.01)
    dD=c[2].slider("δD",0.05,0.25,0.12,0.01)
    dAI=c[3].slider("δAI",0.05,0.30,0.15,0.01)
    if st.button("🚀 Tối ưu động"):
        comp,shares,df,front=b8.compare_strategies(rho,dK,dD,dAI)
        metric_row([("Tỷ lệ K",f"{shares[0]:.1%}",None),("Tỷ lệ D",f"{shares[1]:.1%}",None),("Tỷ lệ AI",f"{shares[2]:.1%}",None),("Tỷ lệ H",f"{shares[3]:.1%}",None)])
        st.plotly_chart(b8.fig_trajectories(df), use_container_width=True)
        st.dataframe(comp, use_container_width=True)
        analysis("ρ cao làm mô hình coi trọng dài hạn hơn, thường tăng tỷ lệ đầu tư có hiệu ứng tích lũy. Front-load có thể tốt nếu lợi ích học hỏi và TFP xuất hiện sớm.")


def page9():
    hero("Bài 9 — Tác động AI tới lao động", "LP phân bổ x_AI và x_H để NetJob không âm cho 8 ngành", ["Labor LP", "NetJob", "Retraining"])
    budget=st.slider("Ngân sách đào tạo + AI",10000,60000,30000,1000)
    if st.button("🚀 Chạy mô phỏng lao động"):
        res,df,total=b9.solve_labor(budget)
        metric_row([("Trạng thái", "Khả thi" if res.success else "Không khả thi",None), ("Tổng NetJob", f"{total:,.0f}",None), ("Ngân sách dùng", f"{(df.x_AI+df.x_H).sum():,.0f}",None)])
        st.plotly_chart(b9.fig_alloc(df), use_container_width=True)
        st.plotly_chart(b9.fig_jobs(df), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        analysis("Ràng buộc Displaced ≤ RetrainingCapacity chính là phát biểu: tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại.")


def page10():
    hero("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn", "First-stage, recourse, VSS và EVPI", ["Stochastic LP", "VSS", "EVPI"])
    cols=st.columns(4)
    probs=[cols[i].slider(p,0.0,1.0,v,0.01) for i,(p,v) in enumerate(zip(["p lạc quan","p cơ sở","p bi quan","p khủng hoảng"],[0.30,0.45,0.20,0.05]))]
    if st.button("🚀 Chạy stochastic programming"):
        xdf,ydf,metrics,ok=b10.solve_sp(probs)
        st.plotly_chart(b10.fig_x(xdf), use_container_width=True)
        st.plotly_chart(b10.fig_metrics(metrics), use_container_width=True)
        st.dataframe(xdf, use_container_width=True)
        st.dataframe(ydf, use_container_width=True)
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        analysis(f"VSS = {metrics['VSS']:,.2f} đo lợi ích của việc xét bất định. EVPI = {metrics['EVPI']:,.2f} là giá trị tối đa của thông tin hoàn hảo.")


def page11():
    hero("Bài 11 — Q-learning cho chính sách kinh tế thích nghi", "MDP 81 trạng thái, 5 hành động, reward phúc lợi xã hội", ["RL tabular", "81 states", "5 actions"])
    c=st.columns(4)
    episodes=c[0].slider("Số episode",500,12000,8000,500)
    alpha=c[1].slider("α learning rate",0.01,0.50,0.10,0.01)
    gamma=c[2].slider("γ discount",0.50,0.99,0.95,0.01)
    seed=c[3].number_input("Seed",0,9999,42,1)
    if st.button("🚀 Train Q-learning"):
        curve,policy,mean,Q=b11.train_q(episodes,alpha,gamma,seed)
        metric_row([("Mean reward 100 ep cuối",f"{mean:.2f}",None),("Rule a1",f"{b11.evaluate_rule(1):.2f}",None),("Rule a3",f"{b11.evaluate_rule(3):.2f}",None),("Random",f"{b11.evaluate_rule(None):.2f}",None)])
        st.plotly_chart(b11.fig_curve(curve), use_container_width=True)
        st.dataframe(policy, use_container_width=True)
        analysis("Q-learning phù hợp để minh họa chính sách thích nghi theo trạng thái. Tuy nhiên, AI chỉ hỗ trợ ra quyết định, không thay thế trách nhiệm chính trị và kiểm định xã hội.")


def page12():
    hero("Bài 12 — AIDEOM-VN Dashboard tích hợp", "Tích hợp 6 module M1–M6 và so sánh 5 kịch bản chính sách", ["Đồ án tích hợp", "5 kịch bản", "Streamlit"])
    tab1,tab2,tab3=st.tabs(["📊 GDP 5 kịch bản", "🕸️ Radar KPI", "📋 Bảng KPI"])
    df,kpi=b12.simulate_scenarios()
    with tab1:
        st.plotly_chart(b12.fig_gdp(df), use_container_width=True)
    with tab2:
        st.plotly_chart(b12.fig_radar(kpi), use_container_width=True)
    with tab3:
        st.dataframe(kpi[["Kịch bản","GDP_index","D","AI","H","Risk_control"]], use_container_width=True)
    analysis("S5 cân bằng thường có hồ sơ KPI ổn định nhất: không cực đoan vào AI, vẫn duy trì đầu tư số hóa và vốn nhân lực để giảm rủi ro triển khai.")


def main() -> None:
    load_css()
    page = sidebar()
    if page.startswith("🏠"): page_home()
    elif "Bài 1" in page: page1()
    elif "Bài 2" in page: page2()
    elif "Bài 3" in page: page3()
    elif "Bài 4" in page: page4()
    elif "Bài 5" in page: page5()
    elif "Bài 6" in page: page6()
    elif "Bài 7" in page: page7()
    elif "Bài 8" in page: page8()
    elif "Bài 9" in page: page9()
    elif "Bài 10" in page: page10()
    elif "Bài 11" in page: page11()
    elif "Bài 12" in page: page12()


if __name__ == "__main__":
    main()
