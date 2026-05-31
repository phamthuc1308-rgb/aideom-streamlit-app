"""AIDEOM-VN Streamlit application.

UX/UI revised version: dark navy dashboard, simple navigation, consistent cards,
clear control/result/analysis blocks for all 12 exercises. Designed for local
execution, GitHub, and Streamlit Cloud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

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
from utils.data_loader import load_macro, load_regions, load_sectors

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    ("🏠 Trang chủ", "Tổng quan", "Bộ dashboard 12 bài"),
    ("🌱 Bài 1 — Cobb-Douglas + AI", "Dễ", "TFP · GDP 2030"),
    ("💰 Bài 2 — LP ngân sách số", "Dễ", "SciPy · PuLP"),
    ("📊 Bài 3 — Priority 10 ngành", "Dễ", "MCDM · Heatmap"),
    ("🗺️ Bài 4 — LP ngành-vùng", "Trung bình", "24 biến · Công bằng"),
    ("🎯 Bài 5 — MIP 15 dự án", "Trung bình", "Binary · CBC"),
    ("🏆 Bài 6 — TOPSIS 6 vùng", "Trung bình", "Entropy · Ranking"),
    ("🌐 Bài 7 — NSGA-II Pareto", "Khá khó", "4 mục tiêu"),
    ("⌛ Bài 8 — Động 2026–2035", "Khá khó", "SLSQP · Welfare"),
    ("👷 Bài 9 — Lao động & AI", "Khá khó", "NetJob · Retraining"),
    ("🎲 Bài 10 — Stochastic SP", "Khó", "VSS · EVPI"),
    ("🤖 Bài 11 — Q-learning RL", "Khó", "MDP · 81 states"),
    ("🇻🇳 Bài 12 — AIDEOM tích hợp", "Đồ án", "M1–M6 · 5 kịch bản"),
]


def load_css() -> None:
    css = APP_DIR / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def html_block(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def chip(text: str, secondary: bool = False) -> str:
    cls = "chip secondary" if secondary else "chip"
    return f'<span class="{cls}">{text}</span>'


def hero(title: str, subtitle: str, chips: Iterable[str] = ()) -> None:
    chips_html = "".join(chip(c, i > 0) for i, c in enumerate(chips))
    html_block(
        f"""
        <div class="aideom-hero">
          <div class="hero-top">{chips_html}</div>
          <div class="aideom-title">{title}</div>
          <div class="aideom-subtitle">{subtitle}</div>
        </div>
        """
    )


def section(title: str) -> None:
    html_block(f'<div class="section-title">{title}</div>')


def intro(text: str) -> None:
    html_block(f'<div class="step-card"><div class="small-muted">{text}</div></div>')


def analysis(text: str) -> None:
    html_block(
        f"""
        <div class="analysis-box">
          <div class="analysis-title">🔍 Tác nhân phân tích kết quả</div>
          <div>{text}</div>
        </div>
        """
    )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)


def sidebar() -> str:
    st.sidebar.markdown("# 🇻🇳 AIDEOM-VN")
    st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")
    labels = [p[0] for p in PAGES]
    page = st.sidebar.radio("Menu", labels, label_visibility="collapsed")
    level = next((p[1] for p in PAGES if p[0] == page), "")
    note = next((p[2] for p in PAGES if p[0] == page), "")
    st.sidebar.markdown("---")
    html_block(
        f"""
        <div class="sidebar-card">
          <b>Trang hiện tại</b><br>
          <span class="sidebar-muted">{level} · {note}</span>
        </div>
        <div class="sidebar-card">
          <b>Dữ liệu</b><br>
          <span class="sidebar-muted">CSV tự nạp trong thư mục <code>data/</code>. Nếu thiếu, app tự tạo dữ liệu mẫu theo đề.</span>
        </div>
        <div class="sidebar-card">
          <b>Stack</b><br>
          <span class="sidebar-muted">Streamlit · Plotly · SciPy · PuLP · CVXPY · Pyomo · pymoo · gymnasium</span>
        </div>
        """
    )
    return page


def page_home() -> None:
    hero(
        "AIDEOM-VN Dashboard",
        "Ứng dụng Streamlit cho bộ 12 bài Mô hình ra quyết định – Phát triển kinh tế Việt Nam trong kỉ nguyên AI.",
        ["12 bài", "Python", "Tối ưu hóa", "Policy dashboard"],
    )
    intro(
        "Giao diện được thiết kế lại theo dạng dashboard tối: sidebar điều hướng bên trái, mỗi bài có mô tả ngắn, "
        "khối điều khiển, biểu đồ Plotly, bảng kết quả và khung phân tích chính sách. Chọn bài ở sidebar để chạy."
    )
    section("📚 Bản đồ 12 bài")
    rows = []
    for i, (name, level, note) in enumerate(PAGES[1:], start=1):
        rows.append({"Bài": i, "Tên": name.split("—", 1)[-1].strip(), "Cấp độ": level, "Trọng tâm": note})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    section("📁 Kiểm tra dữ liệu")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Macro")
        st.dataframe(load_macro().head(), use_container_width=True)
    with c2:
        st.caption("Sectors")
        st.dataframe(load_sectors().head(), use_container_width=True)
    with c3:
        st.caption("Regions")
        st.dataframe(load_regions().head(), use_container_width=True)
    analysis("Bản này kiểm tra đủ 12 bài theo menu. Các bài nặng được cache hoặc mô phỏng tối ưu hóa gọn để chạy được trên Streamlit Cloud.")


def page1() -> None:
    hero("Bài 1 — Cobb-Douglas mở rộng với AI", "Ước lượng TFP, MAPE, phân rã tăng trưởng và dự báo GDP 2030.", ["Cấp độ dễ", "Macro 2020–2025", "Tổng hệ số = 1"])
    intro("Mô hình: Y = A·K^α·L^β·D^γ·AI^δ·H^θ. Các hệ số được chuẩn hóa để tổng bằng 1, giúp thao tác slider dễ hơn mà vẫn giữ lợi suất không đổi theo quy mô.")
    section("🎛️ Bộ điều khiển")
    cols = st.columns(5)
    alpha = cols[0].slider("α — Vốn K", 0.05, 0.70, 0.33, 0.01)
    beta = cols[1].slider("β — Lao động L", 0.05, 0.70, 0.42, 0.01)
    gamma = cols[2].slider("γ — Số hóa D", 0.01, 0.30, 0.10, 0.01)
    delta = cols[3].slider("δ — AI", 0.01, 0.30, 0.08, 0.01)
    theta = cols[4].slider("θ — Nhân lực H", 0.01, 0.30, 0.07, 0.01)
    if st.button("🚀 Chạy mô hình Bài 1"):
        df, contrib, mape, y2030, c = b1.compute_model(alpha, beta, gamma, delta, theta)
        section("📌 Kết quả chính")
        metric_row([("MAPE", f"{mape:.2f}%", None), ("GDP 2030", f"{y2030:,.1f}", "nghìn tỷ VND"), ("Tổng hệ số", f"{sum(c.values()):.2f}", None)])
        st.plotly_chart(b1.fig_tfp(df), use_container_width=True)
        st.plotly_chart(b1.fig_actual_vs_pred(df), use_container_width=True)
        st.plotly_chart(b1.fig_contrib(contrib), use_container_width=True)
        st.dataframe(df[["year", "GDP_trillion_VND", "TFP_A", "Y_hat", "APE_pct"]], use_container_width=True, hide_index=True)
        st.dataframe(contrib, use_container_width=True, hide_index=True)
        analysis(f"MAPE khoảng {mape:.2f}% cho thấy mô hình dùng TFP trung bình bám khá sát dữ liệu lịch sử. Kịch bản 2030 nhấn mạnh D, AI và H chỉ hiệu quả khi đi cùng tăng K, L và TFP.")


def page2() -> None:
    hero("Bài 2 — LP phân bổ ngân sách số", "Tối ưu 4 hạng mục: hạ tầng số, AI/dữ liệu, nhân lực số và R&D.", ["LP", "SciPy", "PuLP shadow price"])
    intro("Người dùng có thể thay đổi ngân sách, sàn từng hạng mục và tỷ trọng công nghệ chiến lược. App giải bằng SciPy và PuLP, sau đó vẽ đường cong Z*(B).")
    section("🎛️ Bộ điều khiển")
    c = st.columns(6)
    budget = c[0].number_input("Ngân sách", 50, 200, 100, 10)
    floors = [c[i + 1].number_input(f"Sàn x{i+1}", 0, 90, v, 5) for i, v in enumerate([25, 15, 20, 10])]
    strategic = c[5].slider("AI+R&D tối thiểu", 0.20, 0.60, 0.35, 0.01)
    if st.button("🚀 Chạy LP Bài 2"):
        res, df, z = b2.solve_scipy(budget, tuple(floors), strategic)
        _, shadow = b2.solve_pulp(budget, tuple(floors), strategic)
        section("📌 Kết quả tối ưu")
        metric_row([("Trạng thái", "Khả thi" if res.success else "Không khả thi", None), ("Z*", f"{z:,.2f}", None), ("Tổng phân bổ", f"{df['Phân bổ'].sum():,.2f}", None)])
        st.plotly_chart(b2.fig_allocation(df), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not shadow.empty:
            st.dataframe(shadow, use_container_width=True, hide_index=True)
        st.plotly_chart(b2.fig_sensitivity(b2.sensitivity_curve(tuple(floors))), use_container_width=True)
        res30, _, z30 = b2.solve_scipy(budget, (floors[0], floors[1], 30, floors[3]), strategic)
        st.info(f"Trường hợp ưu tiên nhân lực x3 ≥ 30: {'khả thi' if res30.success else 'không khả thi'}, Z* = {z30:.2f}")
        analysis("R&D và AI thường hút vốn vì hệ số tác động cao. Shadow price giúp giải thích nếu ngân sách tăng thêm thì GDP gain kỳ vọng tăng thêm bao nhiêu trong vùng nghiệm hiện tại.")


def page3() -> None:
    hero("Bài 3 — Priority Index 10 ngành", "Chuẩn hóa min-max, xếp hạng ưu tiên và phân tích độ nhạy AI readiness.", ["MCDM", "Priority", "10 ngành"])
    intro("Các tiêu chí tốt được chuẩn hóa min-max thuận chiều; Risk là tiêu chí xấu và bị phạt trong công thức Priority.")
    section("🎛️ Trọng số chính sách")
    defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20]
    cols = st.columns(7)
    weights = [cols[i].slider(b3.LABELS[i], 0.0, 0.50, defaults[i], 0.01) for i in range(6)]
    w_risk = cols[6].slider("Risk penalty", 0.0, 0.50, 0.15, 0.01)
    if st.button("🚀 Tính Priority"):
        rank, norm = b3.compute_priority(weights, w_risk)
        section("📌 Xếp hạng và độ nhạy")
        st.plotly_chart(b3.fig_rank(rank), use_container_width=True)
        st.dataframe(rank, use_container_width=True, hide_index=True)
        with st.expander("Xem ma trận chuẩn hóa"):
            st.dataframe(norm, use_container_width=True)
        st.plotly_chart(b3.fig_heatmap(b3.sensitivity_ai()), use_container_width=True)
        st.dataframe(b3.compare_weight_sets().groupby("Bộ trọng số").head(3), use_container_width=True, hide_index=True)
        analysis(f"Ngành đứng đầu hiện tại là {rank.iloc[0]['sector_name_vi']}. Khi thay đổi trọng số AI readiness, top-3 có thể đổi, nên trọng số không chỉ là kỹ thuật mà còn là quyết định chính sách.")


def page4() -> None:
    hero("Bài 4 — LP phân bổ ngân sách ngành-vùng", "24 biến, 6 vùng, 4 hạng mục, có ràng buộc công bằng vùng miền.", ["LP", "Công bằng vùng", "Heatmap"])
    intro("Mô hình tối đa hóa GDP gain nhưng vẫn giữ sàn/trần ngân sách vùng và điều kiện công bằng số hóa D0 + γ·xD.")
    section("🎛️ Bộ điều khiển")
    c1, c2, c3, c4 = st.columns(4)
    lam = c1.slider("λ công bằng", 0.10, 1.00, 0.70, 0.01)
    floor = c2.number_input("Sàn vùng", 0, 10000, 5000, 500)
    cap = c3.number_input("Trần vùng", 6000, 20000, 12000, 500)
    budget = c4.number_input("Tổng ngân sách", 10000, 80000, 50000, 1000)
    if st.button("🚀 Chạy LP vùng"):
        _, df, z, status = b4.solve_lp(lam, floor, cap, budget)
        if status != "Optimal":
            best = b4.find_max_feasible_lambda(floor, cap)
            st.warning(f"λ={lam:.2f} không tối ưu/khả thi. λ lớn nhất tìm được khoảng {best:.2f}.")
        else:
            section("📌 Phân bổ tối ưu")
            metric_row([("Z*", f"{z:,.0f}", None), ("Trạng thái", status, None), ("Tổng phân bổ", f"{df['Phân bổ'].sum():,.0f}", None)])
            st.plotly_chart(b4.fig_heatmap(df), use_container_width=True)
            st.plotly_chart(b4.fig_region_totals(df), use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame([b4.fairness_cost(lam, floor, cap)]), use_container_width=True, hide_index=True)
            analysis("Công bằng vùng tạo chi phí kinh tế đo bằng chênh lệch GDP gain so với mô hình không C5, nhưng giúp tránh tập trung nguồn lực vào vùng đã mạnh.")


def page5() -> None:
    hero("Bài 5 — MIP lựa chọn 15 dự án", "Chọn danh mục dự án tối ưu với ràng buộc nhị phân, tiên quyết và ngân sách đa năm.", ["MIP", "PuLP CBC", "15 dự án"])
    intro("Dự án có ROI cao chưa chắc được chọn nếu vi phạm ràng buộc loại trừ trung tâm dữ liệu, tiên quyết đào tạo hoặc trần ngân sách 2 năm đầu.")
    section("🎛️ Bộ điều khiển")
    c1, c2 = st.columns(2)
    budget = c1.slider("Ngân sách tổng 5 năm", 50000, 110000, 80000, 5000)
    budget12 = c2.slider("Ngân sách năm 1–2", 25000, 60000, 40000, 2500)
    if st.button("🚀 Chạy MIP"):
        df, z, status = b5.solve_mip(budget, budget12)
        section("📌 Danh mục chọn")
        metric_row([("Trạng thái", status, None), ("Tổng NPV", f"{z:,.0f}", None), ("Số dự án", str(int(df["Chọn"].sum())), None)])
        st.plotly_chart(b5.fig_roi(df), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        df100, z100, st100 = b5.solve_mip(100000, budget12)
        st.info(f"Khi ngân sách 100.000: chọn {int(df100['Chọn'].sum())} dự án, Z*={z100:,.0f}, trạng thái={st100}")
        analysis("MIP biến bài toán danh mục dự án thành lựa chọn có/không. Đây là cách phù hợp để mô hình hóa điều kiện bắt buộc và phụ thuộc giữa các dự án công nghệ lớn.")


def page6() -> None:
    hero("Bài 6 — TOPSIS xếp hạng 6 vùng", "Tính hệ số gần gũi C* và so sánh trọng số chuyên gia với Entropy.", ["TOPSIS", "Entropy", "6 vùng"])
    intro("Gini là tiêu chí chi phí, các tiêu chí còn lại là lợi ích. Điểm TOPSIS cao nghĩa là vùng gần phương án lý tưởng hơn.")
    section("🎛️ Trọng số chuyên gia")
    labels = ["GRDP/ng", "FDI", "Digital", "AI", "LĐ ĐT", "R&D", "Internet", "Gini"]
    cols = st.columns(8)
    weights = [cols[i].slider(labels[i], 0.0, 0.50, float(b6.DEFAULT_W[i]), 0.01) for i in range(8)]
    if st.button("🚀 Tính TOPSIS"):
        rank, _ = b6.topsis(weights)
        comp, ew = b6.compare_methods(weights)
        section("📌 Xếp hạng vùng")
        st.plotly_chart(b6.fig_scores(comp), use_container_width=True)
        st.dataframe(rank, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame({"Tiêu chí": labels, "Entropy weight": ew}), use_container_width=True, hide_index=True)
        analysis(f"Vùng dẫn đầu theo trọng số hiện tại là {rank.iloc[0]['region_name_vi']}. Entropy tăng trọng số cho tiêu chí có độ phân tán lớn, vì vậy kết quả có thể khác trọng số chuyên gia.")


def page7() -> None:
    hero("Bài 7 — Pareto đa mục tiêu", "Tập nghiệm thỏa hiệp giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.", ["NSGA-II/Pareto", "TOPSIS", "4 mục tiêu"])
    intro("Để chạy ổn trên Streamlit Cloud, app sinh tập nghiệm Pareto mô phỏng/cache theo cấu trúc bài toán 24 biến; sau đó dùng TOPSIS để chọn nghiệm thỏa hiệp.")
    section("🎛️ Ưu tiên chính sách")
    c = st.columns(5)
    n = c[0].slider("Số nghiệm mô phỏng", 100, 1500, 500, 100)
    weights = [c[i + 1].slider(name, 0.0, 0.80, val, 0.01) for i, (name, val) in enumerate(zip(["w GDP", "w bao trùm", "w môi trường", "w an ninh"], [0.40, 0.25, 0.20, 0.15]))]
    if st.button("🚀 Chạy Pareto"):
        with st.status("Đang sinh và chấm điểm tập nghiệm Pareto...", expanded=False):
            pareto, _ = b7.run_pareto(n)
            scored, best, cost = b7.choose_topsis(pareto, weights)
        section("📌 Tập nghiệm và nghiệm thỏa hiệp")
        metric_row([("Số nghiệm", str(len(pareto)), None), ("Nghiệm chọn", str(int(best.solution_id)), None), ("GDP gain", f"{best.GDP_gain:,.0f}", None)])
        st.plotly_chart(b7.fig_3d(scored), use_container_width=True)
        st.plotly_chart(b7.fig_parallel(scored.head(100)), use_container_width=True)
        st.dataframe(scored.head(20), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([cost]), use_container_width=True, hide_index=True)
        analysis("Pareto không cho một đáp án duy nhất. TOPSIS chỉ là công cụ chọn nghiệm theo trọng số; trọng số vẫn cần được thảo luận và bảo vệ bằng lập luận chính sách.")


def page8() -> None:
    hero("Bài 8 — Tối ưu động 2026–2035", "Tối đa hóa phúc lợi liên thời gian và so sánh chiến lược đầu tư đều/front-load.", ["Dynamic", "SLSQP", "Welfare"])
    intro("App tối ưu 4 tỷ lệ đầu tư K, D, AI, H bằng SLSQP. Kết quả cho thấy quỹ đạo K, D, AI, H, Y và C qua 2026–2035.")
    section("🎛️ Bộ điều khiển động học")
    c = st.columns(4)
    rho = c[0].slider("ρ chiết khấu", 0.80, 0.995, 0.97, 0.005)
    dK = c[1].slider("δK", 0.01, 0.12, 0.05, 0.01)
    dD = c[2].slider("δD", 0.05, 0.25, 0.12, 0.01)
    dAI = c[3].slider("δAI", 0.05, 0.30, 0.15, 0.01)
    if st.button("🚀 Tối ưu động"):
        comp, shares, df, _ = b8.compare_strategies(rho, dK, dD, dAI)
        section("📌 Quỹ đạo tối ưu")
        metric_row([("K", f"{shares[0]:.1%}", None), ("D", f"{shares[1]:.1%}", None), ("AI", f"{shares[2]:.1%}", None), ("H", f"{shares[3]:.1%}", None)])
        st.plotly_chart(b8.fig_trajectories(df), use_container_width=True)
        st.dataframe(comp, use_container_width=True, hide_index=True)
        analysis("Khi ρ cao, mô hình quan tâm dài hạn hơn và có xu hướng ưu tiên các khoản đầu tư tích lũy năng lực. Nếu ρ thấp, tiêu dùng hiện tại và lợi ích ngắn hạn được ưu tiên hơn.")


def page9() -> None:
    hero("Bài 9 — Tác động AI tới lao động", "Phân bổ ngân sách AI và đào tạo lại để NetJob không âm cho 8 ngành.", ["Labor LP", "NetJob", "Retraining"])
    intro("NetJob = việc làm mới do AI + việc làm nâng cấp nhờ đào tạo - việc làm bị thay thế bởi tự động hóa. Ràng buộc retraining bảo đảm không tự động hóa quá nhanh.")
    section("🎛️ Bộ điều khiển")
    budget = st.slider("Ngân sách đào tạo + AI", 10000, 60000, 30000, 1000)
    if st.button("🚀 Chạy mô phỏng lao động"):
        res, df, total = b9.solve_labor(budget)
        section("📌 Kết quả lao động")
        metric_row([("Trạng thái", "Khả thi" if res.success else "Không khả thi", None), ("Tổng NetJob", f"{total:,.0f}", None), ("Ngân sách dùng", f"{(df.x_AI + df.x_H).sum():,.0f}", None)])
        st.plotly_chart(b9.fig_alloc(df), use_container_width=True)
        st.plotly_chart(b9.fig_jobs(df), use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        analysis("Ràng buộc Displaced ≤ RetrainingCapacity thể hiện nguyên tắc: tốc độ tự động hóa không được vượt quá năng lực đào tạo lại và hấp thụ lao động.")


def page10() -> None:
    hero("Bài 10 — Stochastic Programming hai giai đoạn", "First-stage, recourse, VSS và EVPI dưới 4 kịch bản bất định.", ["Stochastic LP", "VSS", "EVPI"])
    intro("Xác suất kịch bản được chuẩn hóa tự động nếu tổng khác 1. VSS đo lợi ích của tư duy xác suất; EVPI đo giá trị thông tin hoàn hảo.")
    section("🎛️ Xác suất kịch bản")
    cols = st.columns(4)
    probs = [cols[i].slider(name, 0.0, 1.0, val, 0.01) for i, (name, val) in enumerate(zip(["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"], [0.30, 0.45, 0.20, 0.05]))]
    if st.button("🚀 Chạy stochastic programming"):
        xdf, ydf, metrics, _ = b10.solve_sp(probs)
        section("📌 Quyết định và giá trị thông tin")
        st.plotly_chart(b10.fig_x(xdf), use_container_width=True)
        st.plotly_chart(b10.fig_metrics(metrics), use_container_width=True)
        st.dataframe(xdf, use_container_width=True, hide_index=True)
        st.dataframe(ydf, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)
        analysis(f"VSS = {metrics['VSS']:,.2f}; EVPI = {metrics['EVPI']:,.2f}. Nếu VSS dương, xét bất định tạo ra giá trị so với quyết định theo kịch bản trung bình.")


def page11() -> None:
    hero("Bài 11 — Q-learning cho chính sách thích nghi", "MDP 81 trạng thái, 5 hành động và reward phúc lợi xã hội.", ["RL tabular", "81 trạng thái", "5 hành động"])
    intro("Giao diện giống mẫu bạn gửi: khối tham số ở trên, nút train, thanh trạng thái, learning curve và bảng chính sách đặc trưng ở dưới.")
    section("🎛️ Tham số huấn luyện")
    c = st.columns(4)
    episodes = c[0].slider("Số episode", 500, 12000, 8000, 500)
    alpha = c[1].slider("α learning rate", 0.01, 0.50, 0.10, 0.01)
    gamma = c[2].slider("γ discount", 0.50, 0.99, 0.95, 0.01)
    seed = c[3].number_input("Seed", 0, 9999, 42, 1)
    if st.button("🚀 Train Q-learning"):
        curve, policy, mean, _ = b11.train_q(episodes, alpha, gamma, seed)
        st.success(f"Done. Mean reward 100 episode cuối: {mean:.2f}")
        section("📈 Learning curve")
        metric_row([("π* mean", f"{mean:.2f}", None), ("Rule a1", f"{b11.evaluate_rule(1):.2f}", None), ("Rule a3", f"{b11.evaluate_rule(3):.2f}", None), ("Random", f"{b11.evaluate_rule(None):.2f}", None)])
        st.plotly_chart(b11.fig_curve(curve), use_container_width=True)
        st.dataframe(policy, use_container_width=True, hide_index=True)
        analysis("Q-learning minh họa chính sách thích nghi theo trạng thái. Kết quả cần được xem là gợi ý phân tích, không thay thế quy trình hoạch định chính sách và trách nhiệm giải trình.")


def page12() -> None:
    hero("Bài 12 — AIDEOM-VN Dashboard tích hợp", "Tích hợp 6 module M1–M6 và so sánh 5 kịch bản chính sách đến 2030.", ["Đồ án tích hợp", "6 module", "5 kịch bản"])
    intro("Trang đồ án được bố cục rõ hơn: tổng quan module, biểu đồ GDP 5 kịch bản, radar KPI và bảng KPI. Các tab giống dashboard mẫu nhưng giảm rối, dễ đọc hơn.")
    modules = [
        ("M1", "Dự báo kinh tế", "Cobb-Douglas, TFP, GDP 2030"),
        ("M2", "Sẵn sàng số", "TOPSIS, Entropy, Digital + AI Index"),
        ("M3", "Tối ưu phân bổ", "LP ngành-vùng và động học"),
        ("M4", "Lao động", "NetJob, retraining capacity"),
        ("M5", "Rủi ro", "Pareto, stochastic, cyber/env"),
        ("M6", "Dashboard", "Streamlit, Plotly, cảnh báo"),
    ]
    section("🧩 6 module AIDEOM-VN")
    cols = st.columns(3)
    for idx, (no, name, desc) in enumerate(modules):
        with cols[idx % 3]:
            html_block(f'<div class="module-card"><div class="module-no">{no}</div><div class="module-name">{name}</div><div class="module-desc">{desc}</div></div>')
    df, kpi = b12.simulate_scenarios()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 GDP 5 kịch bản", "🕸️ Radar KPI", "📋 Bảng KPI", "🔍 Hàm ý"])
    with tab1:
        st.plotly_chart(b12.fig_gdp(df), use_container_width=True)
    with tab2:
        st.plotly_chart(b12.fig_radar(kpi), use_container_width=True)
    with tab3:
        st.dataframe(kpi[["Kịch bản", "GDP_index", "D", "AI", "H", "Risk_control"]], use_container_width=True, hide_index=True)
    with tab4:
        analysis("S5 cân bằng thường có hồ sơ KPI ổn định nhất: không cực đoan vào AI, vẫn duy trì số hóa và vốn nhân lực để giảm rủi ro triển khai. S3 có thể tăng AI nhanh nhưng cần H đi kèm để hạn chế rủi ro xã hội và dữ liệu.")


def main() -> None:
    """Route the selected sidebar label to the correct page.

    Important: use exact page labels / explicit mapping instead of substring checks.
    A substring check such as ``"Bài 1" in page`` also matches "Bài 10",
    "Bài 11" and "Bài 12", which was the cause of pages 10-12 opening
    the Bài 1 screen.
    """
    load_css()
    page = sidebar()

    routes = {
        "🏠 Trang chủ": page_home,
        "🌱 Bài 1 — Cobb-Douglas + AI": page1,
        "💰 Bài 2 — LP ngân sách số": page2,
        "📊 Bài 3 — Priority 10 ngành": page3,
        "🗺️ Bài 4 — LP ngành-vùng": page4,
        "🎯 Bài 5 — MIP 15 dự án": page5,
        "🏆 Bài 6 — TOPSIS 6 vùng": page6,
        "🌐 Bài 7 — NSGA-II Pareto": page7,
        "⌛ Bài 8 — Động 2026–2035": page8,
        "👷 Bài 9 — Lao động & AI": page9,
        "🎲 Bài 10 — Stochastic SP": page10,
        "🤖 Bài 11 — Q-learning RL": page11,
        "🇻🇳 Bài 12 — AIDEOM tích hợp": page12,
    }
    routes.get(page, page_home)()


if __name__ == "__main__":
    main()
