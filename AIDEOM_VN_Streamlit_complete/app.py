from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data import load_default_data, reset_dataframes
from utils.ui import inject_css, banner, analysis_box, caption, TEAL, CORAL, NAVY, GOLD
from utils.charts import apply_theme, line, bar, heatmap, radar, parallel
from utils import models

st.set_page_config(
    page_title='AIDEOM-VN Decision Dashboard',
    page_icon='🇻🇳',
    layout='wide',
    initial_sidebar_state='expanded',
)

inject_css()

# =========================
# Cache cho mô hình nặng
# =========================
@st.cache_data(show_spinner=False)
def cached_pareto(seed: int, n: int) -> pd.DataFrame:
    return models.generate_pareto(seed=seed, n=n)

@st.cache_data(show_spinner=False)
def cached_q_learning(episodes: int, alpha: float, gamma: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return models.q_learning(episodes, alpha, gamma, seed)

@st.cache_data(show_spinner=False)
def cached_dynamic(rho: float, depreciation: float, budget: float) -> pd.DataFrame:
    return models.dynamic_optimization(rho, depreciation, budget)

# =========================
# State dữ liệu
# =========================
if 'macro_df' not in st.session_state:
    macro_df, sectors_df, regions_df = load_default_data()
    st.session_state.macro_df = macro_df
    st.session_state.sectors_df = sectors_df
    st.session_state.regions_df = regions_df

# =========================
# Sidebar navigation
# =========================
st.sidebar.markdown('## 🇻🇳 AIDEOM-VN')
st.sidebar.caption('Dashboard mô hình ra quyết định kinh tế Việt Nam')

with st.sidebar.expander('📚 Dữ liệu & nhóm dễ', expanded=True):
    p0 = st.button('📊 Tổng quan dữ liệu', use_container_width=True)
    p1 = st.button('📈 Bài 1 — Cobb-Douglas', use_container_width=True)
    p2 = st.button('💰 Bài 2 — LP ngân sách', use_container_width=True)
    p3 = st.button('🎯 Bài 3 — Priority ngành', use_container_width=True)
with st.sidebar.expander('⚙️ Tối ưu hóa trung bình', expanded=True):
    p4 = st.button('🗺️ Bài 4 — LP vùng × hạng mục', use_container_width=True)
    p5 = st.button('🧩 Bài 5 — MIP dự án', use_container_width=True)
    p6 = st.button('🏆 Bài 6 — TOPSIS vùng', use_container_width=True)
with st.sidebar.expander('🤖 Mô hình nâng cao', expanded=True):
    p7 = st.button('🌐 Bài 7 — Pareto NSGA-II', use_container_width=True)
    p8 = st.button('⏳ Bài 8 — Tối ưu động', use_container_width=True)
    p9 = st.button('👷 Bài 9 — Lao động & AI', use_container_width=True)
    p10 = st.button('🎲 Bài 10 — Stochastic', use_container_width=True)
    p11 = st.button('🤖 Bài 11 — Q-learning', use_container_width=True)
    p12 = st.button('📡 Bài 12 — 5 kịch bản', use_container_width=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Tổng quan dữ liệu'
for clicked, page_name in [
    (p0, 'Tổng quan dữ liệu'), (p1, 'Bài 1'), (p2, 'Bài 2'), (p3, 'Bài 3'), (p4, 'Bài 4'), (p5, 'Bài 5'), (p6, 'Bài 6'),
    (p7, 'Bài 7'), (p8, 'Bài 8'), (p9, 'Bài 9'), (p10, 'Bài 10'), (p11, 'Bài 11'), (p12, 'Bài 12')
]:
    if clicked:
        st.session_state.page = page_name

st.sidebar.divider()
if st.sidebar.button('🔄 Reset toàn bộ dữ liệu mẫu', use_container_width=True):
    st.session_state.macro_df, st.session_state.sectors_df, st.session_state.regions_df = reset_dataframes()
    st.rerun()

st.sidebar.markdown('''
**Thiết kế:** gradient xanh đậm, card sáng, Plotly tương tác.  
**Dữ liệu:** tự tạo CSV nếu thiếu file.  
**Tương tác:** đổi số liệu và slider → dashboard tự chạy lại.
''')

# =========================
# Helper UI
# =========================
def editable_data_section():
    st.subheader('✏️ Chỉnh dữ liệu đầu vào')
    tab1, tab2, tab3 = st.tabs(['Macro 2020–2025', '10 ngành', '6 vùng'])
    with tab1:
        st.session_state.macro_df = st.data_editor(st.session_state.macro_df, use_container_width=True, num_rows='fixed', key='macro_editor')
    with tab2:
        st.session_state.sectors_df = st.data_editor(st.session_state.sectors_df, use_container_width=True, num_rows='fixed', key='sectors_editor')
    with tab3:
        st.session_state.regions_df = st.data_editor(st.session_state.regions_df, use_container_width=True, num_rows='fixed', key='regions_editor')


def section_result_chart_analysis(result_fn, chart_fn, analysis_fn):
    tab_a, tab_b, tab_c = st.tabs(['① Kết quả số', '② Biểu đồ', '③ Tác nhân phân tích'])
    with tab_a: result_fn()
    with tab_b: chart_fn()
    with tab_c: analysis_fn()

# =========================
# Pages
# =========================
def page_overview():
    banner('Tổng quan dữ liệu AIDEOM-VN', 'Kiểm tra, chỉnh sửa và tải dữ liệu đầu vào cho 12 bài tập mô hình ra quyết định.', '📊')
    macro, sectors, regions = st.session_state.macro_df, st.session_state.sectors_df, st.session_state.regions_df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('GDP 2025', f"{macro['GDP_trillion_VND'].iloc[-1]:,.1f}", 'nghìn tỷ VND')
    c2.metric('Số ngành', len(sectors), 'ngành')
    c3.metric('Số vùng', len(regions), 'vùng')
    c4.metric('Digital/GDP 2025', f"{macro['digital_economy_gdp_pct'].iloc[-1]:.1f}%")
    editable_data_section()
    col1, col2 = st.columns(2)
    with col1:
        fig = line(macro, 'year', ['GDP_trillion_VND', 'capital_stock_trillion_VND'], 'GDP và vốn tích lũy')
        st.plotly_chart(fig, use_container_width=True)
        caption('Biểu đồ đường cho thấy xu hướng GDP và vốn tích lũy theo thời gian.')
    with col2:
        fig = bar(sectors.sort_values('ai_readiness_0_100', ascending=False), 'sector_name_vi', 'ai_readiness_0_100', 'AI readiness theo ngành')
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        caption('Ngành có AI readiness cao thường phù hợp triển khai công nghệ sớm hơn.')
    analysis_box(
        'Dữ liệu gồm 3 bảng: vĩ mô, ngành và vùng.',
        'Các bảng có thể chỉnh trực tiếp bằng data editor; sau khi chỉnh, các bài sẽ tự tính lại.',
        'Cần kiểm tra đơn vị trước khi diễn giải: nghìn tỷ VND, triệu lao động, phần trăm và chỉ số 0–100.',
        'Dữ liệu trong app là dữ liệu giảng dạy, không thay thế số liệu gốc khi viết nghiên cứu học thuật.',
        icon='📌'
    )


def page_1():
    banner('Bài 1 — Cobb-Douglas mở rộng với AI và số hóa', 'Ước lượng TFP, phân rã tăng trưởng và dự báo GDP 2030.', '📈')
    st.markdown('### Tham số co giãn')
    c = st.columns(5)
    a = c[0].slider('α — vốn K', 0.05, 0.70, 0.33, 0.01)
    b = c[1].slider('β — lao động L', 0.05, 0.70, 0.42, 0.01)
    g = c[2].slider('γ — số hóa D', 0.00, 0.30, 0.10, 0.01)
    d = c[3].slider('δ — AI', 0.00, 0.30, 0.08, 0.01)
    t = c[4].slider('θ — nhân lực số H', 0.00, 0.30, 0.07, 0.01)
    s = a+b+g+d+t
    if abs(s-1) > 0.05:
        st.warning(f'Tổng hệ số hiện là {s:.2f}. Theo đề bài nên gần bằng 1 để phản ánh lợi suất không đổi theo quy mô.')
    out, decomp, mape, y2030 = models.cobb_douglas(st.session_state.macro_df, (a,b,g,d,t))
    def result():
        c1, c2, c3 = st.columns(3)
        c1.metric('MAPE dự báo trong mẫu', f'{mape:.2f}%')
        c2.metric('TFP bình quân', f"{out['TFP_A'].mean():.2f}")
        c3.metric('GDP 2030 dự báo', f'{y2030:,.0f}', 'nghìn tỷ VND')
        st.dataframe(out, use_container_width=True)
        st.dataframe(decomp, use_container_width=True)
    def charts():
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(line(out, 'year', 'TFP_A', 'TFP A_t theo năm'), use_container_width=True)
            caption('Đường TFP cho thấy phần tăng trưởng không giải thích trực tiếp bởi K, L, D, AI, H.')
        with c2:
            st.plotly_chart(line(out, 'year', ['GDP_trillion_VND','GDP_hat'], 'GDP thực tế và GDP dự báo'), use_container_width=True)
            caption('Nếu hai đường gần nhau, cấu trúc Cobb-Douglas đang mô tả dữ liệu tương đối tốt.')
        st.plotly_chart(bar(decomp, 'Nguồn', 'Tỷ trọng trong tăng trưởng (%)', 'Phân rã đóng góp tăng trưởng'), use_container_width=True)
    def analysis():
        top = decomp.iloc[decomp['Tỷ trọng trong tăng trưởng (%)'].abs().argmax()]
        analysis_box(
            f'GDP 2030 dự báo khoảng {y2030:,.0f} nghìn tỷ VND; MAPE trong mẫu {mape:.2f}%.',
            f'Thành phần có đóng góp lớn nhất theo trị tuyệt đối là {top["Nguồn"]}.',
            'Chính sách nên ưu tiên yếu tố có đóng góp cao nhưng vẫn duy trì cân bằng giữa vốn, lao động, số hóa và nhân lực số.',
            'Kết quả nhạy với hệ số co giãn; cần kiểm định hoặc hiệu chỉnh bằng dữ liệu dài hơn nếu dùng cho dự báo chính thức.',
        )
    section_result_chart_analysis(result, charts, analysis)


def page_2():
    banner('Bài 2 — LP phân bổ ngân sách 4 hạng mục', 'Tối đa hóa GDP gain kỳ vọng và đọc shadow price.', '💰')
    c = st.columns(5)
    B = c[0].slider('Ngân sách tổng', 70.0, 180.0, 100.0, 5.0)
    f1 = c[1].slider('Sàn hạ tầng', 0.0, 50.0, 25.0, 1.0)
    f2 = c[2].slider('Sàn AI', 0.0, 50.0, 15.0, 1.0)
    f3 = c[3].slider('Sàn nhân lực', 0.0, 60.0, 20.0, 1.0)
    f4 = c[4].slider('Sàn R&D', 0.0, 50.0, 10.0, 1.0)
    alloc, z, shadow = models.lp_budget_4(B, (f1,f2,f3,f4))
    budgets = np.arange(80, 181, 10)
    sens = pd.DataFrame([{'B': bb, 'Z*': models.lp_budget_4(bb, (f1,f2,f3,f4))[1]} for bb in budgets])
    def result():
        st.metric('Z* tối ưu', f'{z:,.2f}', 'nghìn tỷ VND GDP gain')
        st.dataframe(alloc, use_container_width=True)
        st.dataframe(shadow, use_container_width=True)
    def charts():
        c1, c2 = st.columns(2)
        c1.plotly_chart(bar(alloc, 'Hạng mục', 'Ngân sách', 'Phân bổ ngân sách tối ưu'), use_container_width=True)
        c2.plotly_chart(line(sens, 'B', 'Z*', 'Đường cong Z*(B)'), use_container_width=True)
        caption('Z*(B) biểu diễn mức tăng mục tiêu khi ngân sách tổng thay đổi.')
    def analysis():
        main_item = alloc.sort_values('Ngân sách', ascending=False).iloc[0]['Hạng mục'] if np.isfinite(z) else 'không xác định'
        analysis_box(
            f'Hạng mục nhận nhiều ngân sách nhất là {main_item}.',
            'LP dồn phần ngân sách dư vào hạng mục có hệ số tác động biên cao nhất sau khi thỏa các sàn và tỷ lệ công nghệ chiến lược.',
            'Shadow price ngân sách cho biết giá trị tăng thêm của một đơn vị ngân sách nếu ràng buộc ngân sách đang chặt.',
            'Nếu tổng các sàn vượt ngân sách thì bài toán không khả thi.', icon='💡', yellow=True
        )
    section_result_chart_analysis(result, charts, analysis)


def page_3():
    banner('Bài 3 — Priority Index 10 ngành', 'Chuẩn hóa min-max, gán trọng số và phân tích độ nhạy.', '🎯')
    labels = ['Growth','Productivity','Spillover','Export','Employment','AI readiness','Low risk']
    defaults = [0.15,0.15,0.20,0.15,0.10,0.20,0.15]
    cols = st.columns(7)
    w = [cols[i].slider(labels[i], 0.01, 0.50, defaults[i], 0.01) for i in range(7)]
    rank, norm = models.priority_index(st.session_state.sectors_df, w)
    sens_rows = []
    for aiw in np.arange(0.05, 0.41, 0.05):
        ww = defaults.copy(); ww[5] = float(aiw)
        r, _ = models.priority_index(st.session_state.sectors_df, ww)
        for _, row in r.head(5).iterrows():
            sens_rows.append({'AI weight': round(aiw,2), 'Ngành': row['sector_name_vi'], 'Priority': row['Priority']})
    sens = pd.DataFrame(sens_rows)
    def result():
        st.dataframe(rank, use_container_width=True)
        st.dataframe(norm.assign(sector_name_vi=st.session_state.sectors_df['sector_name_vi']), use_container_width=True)
    def charts():
        c1, c2 = st.columns([1,1])
        c1.plotly_chart(bar(rank, 'sector_name_vi', 'Priority', 'Xếp hạng Priority Index'), use_container_width=True)
        pivot = sens.pivot_table(index='Ngành', columns='AI weight', values='Priority', aggfunc='mean').fillna(0)
        c2.plotly_chart(heatmap(pivot.values, pivot.columns.tolist(), pivot.index.tolist(), 'Heatmap độ nhạy theo trọng số AI'), use_container_width=True)
        caption('Heatmap cho thấy ngành nào ổn định trong top khi tăng trọng số AI readiness.')
    def analysis():
        top3 = ', '.join(rank.head(3)['sector_name_vi'].tolist())
        analysis_box(
            f'Top-3 ngành theo bộ trọng số hiện tại: {top3}.',
            'Chỉ số ưu tiên kết hợp tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm, AI readiness và rủi ro tự động hóa thấp.',
            'Có thể dùng kết quả để chọn ngành thí điểm chuyển đổi số trước, sau đó mở rộng sang ngành lan tỏa cao.',
            'Trọng số mang tính chính sách; nên có tham vấn chuyên gia và kiểm tra độ nhạy trước khi ra quyết định.',
        )
    section_result_chart_analysis(result, charts, analysis)


def page_4():
    banner('Bài 4 — LP 6 vùng × 4 hạng mục có công bằng λ', 'Phân bổ ngân sách số theo vùng và hạng mục với ràng buộc công bằng.', '🗺️')
    c = st.columns(4)
    B = c[0].slider('Ngân sách tổng', 30000.0, 80000.0, 50000.0, 1000.0)
    lam = c[1].slider('λ công bằng', 0.50, 0.95, 0.70, 0.01)
    floor = c[2].slider('Sàn mỗi vùng', 1000.0, 10000.0, 5000.0, 500.0)
    cap = c[3].slider('Trần mỗi vùng', 6000.0, 20000.0, 12000.0, 500.0)
    alloc, z = models.lp_region_budget(B, lam, floor, cap)
    def result():
        st.metric('GDP gain Z*', f'{z:,.0f}')
        st.dataframe(alloc, use_container_width=True)
        st.dataframe(alloc.assign(Tổng=alloc[['I','D','AI','H']].sum(axis=1))[['Vùng','Tổng']], use_container_width=True)
    def charts():
        zmat = alloc[['I','D','AI','H']].to_numpy()
        st.plotly_chart(heatmap(zmat, ['I','D','AI','H'], alloc['Vùng'].tolist(), 'Heatmap phân bổ ngân sách vùng × hạng mục'), use_container_width=True)
        caption('Màu đậm biểu thị mức phân bổ cao hơn cho vùng và hạng mục tương ứng.')
    def analysis():
        top_region = alloc.assign(Tổng=alloc[['I','D','AI','H']].sum(axis=1)).sort_values('Tổng', ascending=False).iloc[0]['Vùng']
        analysis_box(
            f'Vùng nhận nhiều ngân sách nhất là {top_region}.',
            'λ càng cao thì mô hình càng buộc các vùng có mức số hóa thấp phải được cải thiện tương đối so với vùng dẫn đầu.',
            'Ràng buộc công bằng giúp tránh tập trung toàn bộ vốn vào vùng đã sẵn sàng số cao.',
            'Công bằng có thể làm giảm Z*; cần cân đối giữa hiệu quả kinh tế ngắn hạn và hội tụ vùng dài hạn.',
            icon='📌'
        )
    section_result_chart_analysis(result, charts, analysis)


def page_5():
    banner('Bài 5 — MIP chọn 15 dự án chuyển đổi số', 'Knapsack có ràng buộc loại trừ, tiên quyết và số lượng dự án.', '🧩')
    B = st.slider('Ngân sách 5 năm', 50000.0, 120000.0, 80000.0, 5000.0)
    selected, z, cost = models.project_mip(B)
    def result():
        c1, c2, c3 = st.columns(3)
        c1.metric('Tổng NPV', f'{z:,.0f}')
        c2.metric('Tổng chi phí chọn', f'{cost:,.0f}')
        c3.metric('Số dự án chọn', int((selected['Chọn']=='Có').sum()))
        st.dataframe(selected, use_container_width=True)
    def charts():
        st.plotly_chart(bar(selected, 'Mã', 'NPV', 'NPV dự án', color='Chọn'), use_container_width=True)
        caption('Cột màu cho biết dự án được chọn trong nghiệm MIP.')
    def analysis():
        chosen = selected[selected['Chọn']=='Có']['Mã'].tolist()
        analysis_box(
            f'Mô hình chọn {len(chosen)} dự án: {", ".join(chosen)}.',
            'MIP tối đa hóa NPV trong khi thỏa ngân sách, ngân sách giai đoạn đầu, loại trừ trung tâm dữ liệu và các quan hệ tiên quyết.',
            'Danh mục chọn giúp ưu tiên dự án có lợi ích cao nhưng vẫn bảo đảm an ninh mạng và chính phủ số tối thiểu.',
            'Mô hình chưa tính cộng hưởng giữa các dự án; có thể bổ sung biến nhị phân tương tác nếu cần.', icon='🎯', yellow=True
        )
    section_result_chart_analysis(result, charts, analysis)


def page_6():
    banner('Bài 6 — TOPSIS xếp hạng 6 vùng', 'So sánh trọng số chuyên gia với trọng số entropy.', '🏆')
    labels = ['GRDP/người','FDI','Digital','AI','LĐ đào tạo','R&D','Internet','Gini thấp']
    defaults = [0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10]
    cols = st.columns(4)
    weights = []
    for i, lab in enumerate(labels):
        weights.append(cols[i%4].slider(lab, 0.01, 0.50, defaults[i], 0.01, key=f'topsis_{i}'))
    rank, entropy = models.topsis_regions(st.session_state.regions_df, weights)
    def result():
        st.dataframe(rank, use_container_width=True)
        st.dataframe(entropy, use_container_width=True)
    def charts():
        c1, c2 = st.columns(2)
        c1.plotly_chart(bar(rank, 'region_name_vi', 'TOPSIS', 'Điểm TOPSIS theo vùng'), use_container_width=True)
        c2.plotly_chart(bar(entropy, 'Tiêu chí', 'Trọng số entropy', 'Trọng số khách quan entropy'), use_container_width=True)
        caption('TOPSIS chọn vùng gần nghiệm lý tưởng dương và xa nghiệm lý tưởng âm nhất.')
    def analysis():
        top = rank.iloc[0]['region_name_vi']
        analysis_box(
            f'Vùng đứng đầu theo trọng số hiện tại là {top}.',
            'Điểm TOPSIS cao nghĩa là vùng có tổ hợp tốt về GRDP/người, FDI, số hóa, AI readiness, nhân lực, R&D, Internet và Gini thấp.',
            'Kết quả hỗ trợ lựa chọn vùng triển khai trung tâm AI hoặc sandbox dữ liệu trước.',
            'TOPSIS giả định các tiêu chí độc lập; nếu Digital và Internet tương quan cao, cần kiểm tra đa cộng tuyến tiêu chí.',
        )
    section_result_chart_analysis(result, charts, analysis)


def page_7():
    banner('Bài 7 — Tối ưu đa mục tiêu Pareto', 'Mô phỏng nghiệm Pareto 4 mục tiêu và chọn nghiệm thỏa hiệp bằng TOPSIS proxy.', '🌐')
    c = st.columns(3)
    seed = c[0].number_input('Seed', 1, 999, 42)
    n = c[1].slider('Số nghiệm mẫu Pareto', 80, 600, 220, 20)
    run = c[2].button('🚀 Chạy / cập nhật Pareto', use_container_width=True)
    if run:
        prog = st.progress(0, text='Đang mô phỏng Pareto...')
        for i in range(5):
            time.sleep(0.05); prog.progress((i+1)*20, text='Đang mô phỏng Pareto...')
        prog.empty()
    df = cached_pareto(int(seed), int(n))
    def result():
        st.dataframe(df.head(20), use_container_width=True)
        st.metric('Compromise score tốt nhất', f"{df['Compromise'].iloc[0]:.3f}")
    def charts():
        fig = px.scatter_3d(df, x='Tăng trưởng', y='Bao trùm', z='Môi trường', color='An ninh dữ liệu', size='Compromise', hover_data=['Hạ tầng','AI','Nhân lực','Xanh'], title='Biên Pareto 3D')
        st.plotly_chart(apply_theme(fig, 610), use_container_width=True)
        st.plotly_chart(parallel(df.head(80), ['Tăng trưởng','Bao trùm','Môi trường','An ninh dữ liệu','Hạ tầng','AI','Nhân lực','Xanh'], 'Compromise', 'Parallel coordinates nghiệm Pareto'), use_container_width=True)
        caption('Mỗi đường là một phương án chính sách; đường cân bằng cao ở nhiều trục thường là nghiệm thỏa hiệp tốt.')
    def analysis():
        best = df.iloc[0]
        analysis_box(
            f'Nghiệm thỏa hiệp phân bổ xấp xỉ: Hạ tầng {best["Hạ tầng"]:.2f}, AI {best["AI"]:.2f}, Nhân lực {best["Nhân lực"]:.2f}, Xanh {best["Xanh"]:.2f}.',
            'Không có một nghiệm tối ưu tuyệt đối cho mọi mục tiêu; Pareto thể hiện đánh đổi giữa tăng trưởng, bao trùm, môi trường và an ninh.',
            'Nên chọn nghiệm thỏa hiệp sau khi hội đồng chính sách thống nhất trọng số hoặc ngưỡng tối thiểu cho từng mục tiêu.',
            'Đây là mô phỏng nhẹ để chạy ổn định trên Streamlit; khi có pymoo có thể thay bằng NSGA-II đầy đủ.', icon='💡'
        )
    section_result_chart_analysis(result, charts, analysis)


def page_8():
    banner('Bài 8 — Tối ưu động 2026–2035', 'Tìm quỹ đạo đầu tư số tối ưu theo tỷ lệ chiết khấu và khấu hao.', '⏳')
    c = st.columns(3)
    rho = c[0].slider('ρ — tỷ lệ chiết khấu', 0.01, 0.20, 0.06, 0.01)
    dep = c[1].slider('Khấu hao năng lực số', 0.01, 0.20, 0.07, 0.01)
    budget = c[2].slider('Ngân sách giai đoạn', 3000.0, 15000.0, 8000.0, 500.0)
    df = cached_dynamic(rho, dep, budget)
    def result():
        st.dataframe(df, use_container_width=True)
        st.metric('Tổng đầu tư', f"{df['Đầu tư tối ưu'].sum():,.0f}")
    def charts():
        c1, c2 = st.columns(2)
        c1.plotly_chart(line(df, 'Năm', 'Đầu tư tối ưu', 'Quỹ đạo đầu tư tối ưu'), use_container_width=True)
        c2.plotly_chart(line(df, 'Năm', 'Chỉ số năng lực số', 'Tích lũy năng lực số'), use_container_width=True)
        caption('ρ cao thường làm mô hình ưu tiên lợi ích gần hiện tại hơn.')
    def analysis():
        peak = int(df.iloc[df['Đầu tư tối ưu'].idxmax()]['Năm'])
        analysis_box(
            f'Năm có đầu tư cao nhất là {peak}.',
            'Tối ưu động cân bằng giữa lợi ích tích lũy của năng lực số và chi phí cơ hội theo thời gian.',
            'Nếu khấu hao cao, cần đầu tư duy trì thường xuyên; nếu chiết khấu cao, đầu tư sớm có xu hướng hấp dẫn hơn.',
            'Mô hình là dạng proxy; tham số cần hiệu chỉnh bằng dữ liệu ngân sách và hiệu quả đầu tư thực tế.',
        )
    section_result_chart_analysis(result, charts, analysis)


def page_9():
    banner('Bài 9 — Tác động AI tới lao động', 'Ước lượng Jobs displaced, Jobs created và NetJob theo ngành.', '👷')
    budget = st.slider('Ngân sách đào tạo / chuyển đổi lao động', 5.0, 80.0, 30.0, 5.0)
    df = models.labor_ai_impact(st.session_state.sectors_df, budget)
    def result():
        st.dataframe(df, use_container_width=True)
        st.metric('NetJob toàn bộ', f"{df['NetJob_million'].sum():.3f}", 'triệu việc làm')
    def charts():
        long = df.melt(id_vars='sector_name_vi', value_vars=['Jobs_displaced_million','Jobs_created_million','NetJob_million'], var_name='Chỉ tiêu', value_name='Triệu việc làm')
        fig = px.bar(long, x='sector_name_vi', y='Triệu việc làm', color='Chỉ tiêu', barmode='group', title='Tác động AI tới lao động theo ngành')
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(apply_theme(fig, 520), use_container_width=True)
        caption('NetJob âm cho thấy ngành cần chính sách đào tạo lại hoặc bảo hiểm chuyển đổi mạnh hơn.')
    def analysis():
        worst = df.sort_values('NetJob_million').iloc[0]['sector_name_vi']
        analysis_box(
            f'Ngành rủi ro NetJob thấp nhất là {worst}.',
            'Tự động hóa có thể thay thế một phần lao động, trong khi đầu tư AI và số hóa tạo công việc mới.',
            'Cần gắn triển khai AI với đào tạo lại, dịch chuyển kỹ năng và hỗ trợ nhóm lao động dễ bị thay thế.',
            'Ước lượng NetJob phụ thuộc mạnh vào giả định về tốc độ tự động hóa và khả năng tạo việc làm mới.', icon='⚠️', yellow=True
        )
    section_result_chart_analysis(result, charts, analysis)


def page_10():
    banner('Bài 10 — Stochastic programming 2 giai đoạn', 'Tính phân bổ first-stage, VSS và EVPI theo xác suất kịch bản.', '🎲')
    c = st.columns(4)
    B = c[0].slider('Ngân sách', 50.0, 180.0, 100.0, 5.0, key='sp_budget')
    p1 = c[1].slider('Xác suất lạc quan', 0.05, 0.90, 0.25, 0.05)
    p2 = c[2].slider('Xác suất cơ sở', 0.05, 0.90, 0.50, 0.05)
    p3 = c[3].slider('Xác suất bất lợi', 0.05, 0.90, 0.25, 0.05)
    df, vss, evpi = models.stochastic_program(B, (p1,p2,p3))
    def result():
        c1, c2 = st.columns(2)
        c1.metric('VSS', f'{vss:.3f}')
        c2.metric('EVPI', f'{evpi:.3f}')
        st.dataframe(df, use_container_width=True)
    def charts():
        st.plotly_chart(bar(df, 'Hạng mục', 'Phân bổ first-stage', 'Phân bổ first-stage trong stochastic LP'), use_container_width=True)
        scen = pd.DataFrame({'Kịch bản':['Lạc quan','Cơ sở','Bất lợi'], 'Xác suất':np.array([p1,p2,p3]) / (p1+p2+p3)})
        st.plotly_chart(bar(scen, 'Kịch bản', 'Xác suất', 'Xác suất kịch bản sau chuẩn hóa'), use_container_width=True)
        caption('EVPI đo giá trị tối đa của thông tin hoàn hảo; VSS đo lợi ích của mô hình ngẫu nhiên so với quyết định kỳ vọng đơn giản.')
    def analysis():
        analysis_box(
            f'EVPI = {evpi:.3f}, VSS = {vss:.3f}.',
            'EVPI cao nghĩa là thông tin về kịch bản tương lai có giá trị; VSS cao nghĩa là đáng dùng stochastic programming thay vì tối ưu một kịch bản trung bình.',
            'Khi bất định lớn, chính sách nên giữ danh mục đầu tư linh hoạt và có phương án điều chỉnh sau khi quan sát kịch bản.',
            'Xác suất kịch bản là giả định; nên cập nhật bằng dữ liệu thực tế hoặc ý kiến chuyên gia.', icon='📌'
        )
    section_result_chart_analysis(result, charts, analysis)


def page_11():
    banner('Bài 11 — Q-learning 81 trạng thái', 'Huấn luyện chính sách thích nghi với 5 hành động.', '🤖')
    c = st.columns(4)
    episodes = c[0].slider('Episodes', 200, 6000, 2000, 200)
    alpha = c[1].slider('α learning rate', 0.01, 0.50, 0.10, 0.01)
    gamma = c[2].slider('γ discount', 0.50, 0.99, 0.95, 0.01)
    seed = c[3].number_input('Seed', 1, 999, 42)
    if st.button('🚀 Train Q-learning', use_container_width=True):
        prog = st.progress(0, text='Đang huấn luyện Q-learning...')
        for i in range(10):
            time.sleep(0.04); prog.progress((i+1)*10, text='Đang huấn luyện Q-learning...')
        prog.empty()
    curve, pol = cached_q_learning(int(episodes), float(alpha), float(gamma), int(seed))
    action_map = {0:'Truyền thống',1:'Cân bằng',2:'Số hóa nhanh',3:'AI dẫn dắt',4:'Bao trùm'}
    pol['Best_Action_Name'] = pol['Best_Action'].map(action_map)
    def result():
        st.metric('Mean reward 100 episode cuối', f"{curve['Reward'].tail(100).mean():.2f}")
        st.dataframe(pol.head(30), use_container_width=True)
        st.dataframe(pol['Best_Action_Name'].value_counts().rename_axis('Hành động').reset_index(name='Số trạng thái'), use_container_width=True)
    def charts():
        st.plotly_chart(line(curve, 'Episode', ['Reward','Reward_smooth'], 'Learning curve'), use_container_width=True)
        counts = pol['Best_Action_Name'].value_counts().reset_index(); counts.columns=['Hành động','Số trạng thái']
        st.plotly_chart(bar(counts, 'Hành động', 'Số trạng thái', 'Phân bố hành động tối ưu theo 81 trạng thái'), use_container_width=True)
        caption('Đường Reward_smooth giúp quan sát xu hướng học thay vì nhiễu từng episode.')
    def analysis():
        best_action = pol['Best_Action_Name'].value_counts().idxmax()
        analysis_box(
            f'Hành động được chọn nhiều nhất là {best_action}.',
            'Q-learning học chính sách dựa trên phần thưởng gồm tăng trưởng, thất nghiệp, rủi ro mạng và môi trường.',
            'Chính sách thích nghi cho phép đổi hành động khi trạng thái kinh tế thay đổi, thay vì cố định một chiến lược cho mọi tình huống.',
            'Reward còn nhiễu nếu episode ít; tăng episode để chính sách ổn định hơn nhưng thời gian chạy dài hơn.', icon='🤖'
        )
    section_result_chart_analysis(result, charts, analysis)


def page_12():
    banner('Bài 12 — Tổng hợp 5 kịch bản AIDEOM-VN', 'So sánh S1–S5 bằng bảng, cột và radar chart.', '📡')
    df = models.scenarios_summary()
    selected = st.selectbox('Chọn kịch bản để phân tích sâu', df['Kịch bản'].tolist(), index=4)
    def result():
        st.dataframe(df, use_container_width=True)
        row = df[df['Kịch bản']==selected].iloc[0]
        c = st.columns(5)
        for i, col in enumerate(['Tăng trưởng','Bao trùm','Môi trường','An ninh','AI Readiness']):
            c[i].metric(col, f'{row[col]:.2f}')
    def charts():
        cols = ['Tăng trưởng','Bao trùm','Môi trường','An ninh','AI Readiness']
        long = df.melt(id_vars='Kịch bản', value_vars=cols, var_name='Tiêu chí', value_name='Điểm')
        st.plotly_chart(px.bar(long, x='Kịch bản', y='Điểm', color='Tiêu chí', barmode='group', title='So sánh 5 kịch bản S1–S5').update_layout(template='plotly_white'), use_container_width=True)
        series = {r['Kịch bản']: [float(r[c]) for c in cols] for _, r in df.iterrows() if r['Kịch bản'] in [selected, 'S1 Truyền thống', 'S5 Bao trùm + an toàn']}
        st.plotly_chart(radar(cols, series, 'Radar chart kịch bản'), use_container_width=True)
        caption('Radar chart giúp thấy kịch bản cân bằng hay lệch về một mục tiêu.')
    def analysis():
        row = df[df['Kịch bản']==selected].iloc[0]
        best_dim = row[['Tăng trưởng','Bao trùm','Môi trường','An ninh','AI Readiness']].astype(float).idxmax()
        analysis_box(
            f'{selected} mạnh nhất ở tiêu chí {best_dim}.',
            'S5 thường là phương án cân bằng hơn vì không chỉ tối đa tăng trưởng mà còn chú ý bao trùm, môi trường và an ninh.',
            'Có thể dùng S5 làm kịch bản khuyến nghị nếu mục tiêu là phát triển số an toàn và bền vững.',
            'Nếu ưu tiên tăng trưởng tuyệt đối, kịch bản AI dẫn dắt có thể hấp dẫn hơn nhưng cần kiểm soát rủi ro xã hội.', icon='💡', yellow=True
        )
    section_result_chart_analysis(result, charts, analysis)

pages = {
    'Tổng quan dữ liệu': page_overview,
    'Bài 1': page_1,
    'Bài 2': page_2,
    'Bài 3': page_3,
    'Bài 4': page_4,
    'Bài 5': page_5,
    'Bài 6': page_6,
    'Bài 7': page_7,
    'Bài 8': page_8,
    'Bài 9': page_9,
    'Bài 10': page_10,
    'Bài 11': page_11,
    'Bài 12': page_12,
}

pages[st.session_state.page]()
