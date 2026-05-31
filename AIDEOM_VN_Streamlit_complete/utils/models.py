from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
import pulp

# ---------- Bài 1 ----------
def cobb_douglas(macro: pd.DataFrame, weights: tuple[float, float, float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    a, b, g, d, t = weights
    df = macro.copy()
    K = df['capital_stock_trillion_VND'].to_numpy(float)
    L = df['labor_million'].to_numpy(float)
    D = df['digital_economy_gdp_pct'].to_numpy(float)
    AI = df['ai_firms_thousand'].to_numpy(float)
    H = df['trained_labor_pct'].to_numpy(float)
    Y = df['GDP_trillion_VND'].to_numpy(float)
    A = Y / (K**a * L**b * D**g * AI**d * H**t)
    A_mean = float(np.mean(A))
    y_hat = A_mean * K**a * L**b * D**g * AI**d * H**t
    mape = float(np.mean(np.abs((Y - y_hat) / Y)) * 100)
    out = df[['year', 'GDP_trillion_VND']].copy()
    out['TFP_A'] = A
    out['GDP_hat'] = y_hat
    # growth accounting average contributions
    log_vars = {
        'TFP': np.log(A), 'K': np.log(K), 'L': np.log(L), 'D': np.log(D), 'AI': np.log(AI), 'H': np.log(H), 'Y': np.log(Y)
    }
    contribs = {
        'TFP': np.diff(log_vars['TFP']),
        'K': a * np.diff(log_vars['K']),
        'L': b * np.diff(log_vars['L']),
        'D': g * np.diff(log_vars['D']),
        'AI': d * np.diff(log_vars['AI']),
        'H': t * np.diff(log_vars['H']),
    }
    avg_total = float(np.mean(np.diff(log_vars['Y'])))
    rows = []
    for k, v in contribs.items():
        rows.append({'Nguồn': k, 'Đóng góp log bình quân': float(np.mean(v)), 'Tỷ trọng trong tăng trưởng (%)': float(np.mean(v) / avg_total * 100)})
    decomp = pd.DataFrame(rows)
    # 2030 forecast
    last = df.iloc[-1]
    A_2030 = A[-1] * (1.012**5)
    Y2030 = A_2030 * (last['capital_stock_trillion_VND'] * 1.06**5)**a * (last['labor_million'] * 1.06**5)**b * 30**g * 100**d * 35**t
    return out, decomp, mape, float(Y2030)

# ---------- Bài 2 ----------
def lp_budget_4(total_budget: float, floors: tuple[float, float, float, float]) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    c = np.array([-0.85, -1.20, -0.95, -1.35])
    A_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ]
    b_ub = [total_budget, -floors[0], -floors[1], -floors[2], -floors[3], 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method='highs')
    items = ['Hạ tầng số', 'AI & dữ liệu', 'Nhân lực số', 'R&D công nghệ']
    if not res.success:
        alloc = pd.DataFrame({'Hạng mục': items, 'Ngân sách': [np.nan]*4})
        shadow = pd.DataFrame({'Ràng buộc': ['Không khả thi'], 'Shadow price xấp xỉ': [np.nan]})
        return alloc, np.nan, shadow
    x = res.x
    value = -float(res.fun)
    alloc = pd.DataFrame({'Hạng mục': items, 'Ngân sách': x, 'Hệ số tác động': [0.85, 1.20, 0.95, 1.35], 'GDP gain': x * np.array([0.85, 1.20, 0.95, 1.35])})
    # Highs marginal for minimization; convert sign for max objective shadow approximation.
    marg = getattr(res, 'ineqlin', {}).get('marginals', [np.nan]*6) if hasattr(res, 'ineqlin') else [np.nan]*6
    shadow = pd.DataFrame({'Ràng buộc': ['Tổng ngân sách', 'Sàn I', 'Sàn AI', 'Sàn H', 'Sàn R&D', 'Tỷ trọng AI+R&D'], 'Shadow price xấp xỉ': -np.array(marg, dtype=float)})
    return alloc, value, shadow

# ---------- Bài 3 ----------
def priority_index(sectors: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ['growth_rate_2024_pct', 'productivity_million_vnd_per_worker', 'spillover_coef_0_1', 'export_billion_USD', 'labor_million', 'ai_readiness_0_100', 'automation_risk_pct']
    X = sectors[cols].astype(float).copy()
    norm = pd.DataFrame(index=sectors.index)
    for c in cols[:-1]:
        norm[c] = (X[c] - X[c].min()) / (X[c].max() - X[c].min() + 1e-12)
    # Risk is a cost criterion: lower risk is better.
    norm['automation_risk_pct'] = (X['automation_risk_pct'].max() - X['automation_risk_pct']) / (X['automation_risk_pct'].max() - X['automation_risk_pct'].min() + 1e-12)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    score = norm.values @ w
    out = sectors[['sector_name_vi']].copy()
    out['Priority'] = score
    out['Rank'] = out['Priority'].rank(ascending=False, method='dense').astype(int)
    return out.sort_values('Priority', ascending=False), norm

# ---------- Bài 4 ----------
def lp_region_budget(total_budget: float, lam: float, floor_region: float, cap_region: float) -> tuple[pd.DataFrame, float]:
    regions = ['Trung du miền núi phía Bắc', 'Đồng bằng sông Hồng', 'Bắc Trung Bộ + DH Trung Bộ', 'Tây Nguyên', 'Đông Nam Bộ', 'Đồng bằng sông Cửu Long']
    codes = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items = ['I', 'D', 'AI', 'H']
    beta = {
        ('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
        ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
        ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
        ('CH','I'):1.20,('CH','D'):0.75,('CH','AI'):0.45,('CH','H'):1.35,
        ('SE','I'):0.90,('SE','D'):1.30,('SE','AI'):1.55,('SE','H'):1.00,
        ('MD','I'):1.10,('MD','D'):0.85,('MD','AI'):0.65,('MD','H'):1.25,
    }
    D0 = {'NMM':38,'RRD':78,'NCC':55,'CH':32,'SE':82,'MD':48}
    gamma = 0.002
    m = pulp.LpProblem('VN_Digital_Budget', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (codes, items), lowBound=0)
    M = pulp.LpVariable('Dmax', lowBound=0)
    m += pulp.lpSum(beta[(r,j)] * x[r][j] for r in codes for j in items)
    m += pulp.lpSum(x[r][j] for r in codes for j in items) <= total_budget
    for r in codes:
        m += pulp.lpSum(x[r][j] for j in items) >= floor_region
        m += pulp.lpSum(x[r][j] for j in items) <= cap_region
    m += pulp.lpSum(x[r]['H'] for r in codes) >= 0.24 * total_budget
    for r in codes:
        m += D0[r] + gamma * x[r]['D'] <= M
        m += D0[r] + gamma * x[r]['D'] >= lam * M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    rows = []
    for code, name in zip(codes, regions):
        row = {'Vùng': name}
        for j in items:
            row[j] = float(x[code][j].value() or 0)
        rows.append(row)
    alloc = pd.DataFrame(rows)
    return alloc, float(pulp.value(m.objective) or 0)

# ---------- Bài 5 ----------
def project_mip(total_budget: float) -> tuple[pd.DataFrame, float, float]:
    P = list(range(1,16))
    names = {
        1:'TT dữ liệu Hòa Lạc',2:'TT dữ liệu phía Nam',3:'5G toàn quốc',4:'VNeID 2.0',5:'Dịch vụ công v3',6:'Y tế số quốc gia',7:'Giáo dục số K-12',8:'Trung tâm AI quốc gia',9:'Sandbox fintech',10:'Logistics thông minh',11:'Nông nghiệp số ĐBSCL',12:'50.000 kỹ sư AI',13:'KCN bán dẫn Bắc Ninh-BG',14:'SOC an ninh mạng',15:'Open Data quốc gia'
    }
    C = {1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1 = {1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B = {1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    m = pulp.LpProblem('VN_Project_Selection', pulp.LpMaximize)
    y = pulp.LpVariable.dicts('y', P, cat='Binary')
    m += pulp.lpSum(B[i]*y[i] for i in P)
    m += pulp.lpSum(C[i]*y[i] for i in P) <= total_budget
    m += pulp.lpSum(C1[i]*y[i] for i in P) <= 0.50 * total_budget
    m += y[1] + y[2] <= 1
    m += y[8] <= y[12]
    m += y[13] <= y[12]
    m += y[4] + y[5] >= 1
    m += y[14] >= 1
    m += pulp.lpSum(y[i] for i in P) >= 7
    m += pulp.lpSum(y[i] for i in P) <= 11
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    rows, cost = [], 0.0
    for i in P:
        chosen = float(y[i].value() or 0) > .5
        if chosen:
            cost += C[i]
        rows.append({'Mã': f'P{i}', 'Dự án': names[i], 'Chi phí': C[i], 'NPV': B[i], 'Chọn': 'Có' if chosen else 'Không'})
    return pd.DataFrame(rows), float(pulp.value(m.objective) or 0), cost

# ---------- Bài 6 ----------
def topsis_regions(regions: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    criteria = ['grdp_per_capita_million_VND','fdi_registered_billion_USD','digital_index_0_100','ai_readiness_0_100','trained_labor_pct','rd_intensity_pct','internet_penetration_pct','gini_coef']
    X = regions[criteria].astype(float).to_numpy()
    w = np.array(weights, dtype=float); w = w / w.sum()
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R * w
    benefit = np.array([True, True, True, True, True, True, True, False])
    A_pos = np.where(benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit, V.min(axis=0), V.max(axis=0))
    S_pos = np.sqrt(((V - A_pos)**2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    C = S_neg / (S_pos + S_neg + 1e-12)
    out = regions[['region_name_vi']].copy(); out['TOPSIS'] = C; out['Rank'] = out['TOPSIS'].rank(ascending=False, method='dense').astype(int)
    # entropy weights from min-max benefit/cost normalization
    Xn = pd.DataFrame(X, columns=criteria)
    for c in criteria[:-1]: Xn[c] = (Xn[c] - Xn[c].min()) / (Xn[c].max() - Xn[c].min() + 1e-12)
    Xn['gini_coef'] = (Xn['gini_coef'].max() - Xn['gini_coef']) / (Xn['gini_coef'].max() - Xn['gini_coef'].min() + 1e-12)
    P = Xn.to_numpy() / (Xn.to_numpy().sum(axis=0) + 1e-12)
    E = -1 / np.log(len(Xn)) * np.nansum(P * np.log(P + 1e-12), axis=0)
    ew = (1 - E) / (1 - E).sum()
    ent = pd.DataFrame({'Tiêu chí': criteria, 'Trọng số entropy': ew})
    return out.sort_values('TOPSIS', ascending=False), ent

# ---------- Bài 7 ----------
def generate_pareto(seed: int = 42, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(4), size=n)
    growth = 0.35*X[:,0] + 0.30*X[:,1] + 0.18*X[:,2] + 0.17*X[:,3] + rng.normal(0, .015, n)
    inclusion = 0.15*X[:,0] + 0.12*X[:,1] + 0.42*X[:,2] + 0.31*X[:,3] + rng.normal(0, .015, n)
    environment = 0.20*X[:,0] + 0.15*X[:,1] + 0.28*X[:,2] + 0.37*X[:,3] + rng.normal(0, .015, n)
    security = 0.16*X[:,0] + 0.34*X[:,1] + 0.18*X[:,2] + 0.32*X[:,3] + rng.normal(0, .015, n)
    df = pd.DataFrame(X, columns=['Hạ tầng', 'AI', 'Nhân lực', 'Xanh'])
    df['Tăng trưởng'] = growth; df['Bao trùm'] = inclusion; df['Môi trường'] = environment; df['An ninh dữ liệu'] = security
    df['Compromise'] = df[['Tăng trưởng','Bao trùm','Môi trường','An ninh dữ liệu']].mean(axis=1)
    return df.sort_values('Compromise', ascending=False).reset_index(drop=True)

# ---------- Bài 8 ----------
def dynamic_optimization(rho: float, depreciation: float, budget: float = 8000) -> pd.DataFrame:
    years = np.arange(2026, 2036)
    def objective(u):
        state = 100.0; val = 0.0
        for k, spend in enumerate(u):
            state = (1 - depreciation) * state + 0.018 * spend
            benefit = 0.55 * np.log1p(state) + 0.00022 * spend
            val += benefit / ((1 + rho) ** k)
        return -val
    cons = ({'type':'ineq', 'fun': lambda u: budget - np.sum(u)})
    res = minimize(objective, x0=np.ones(10)*budget/10, bounds=[(0, budget)]*10, constraints=cons, method='SLSQP')
    u = res.x if res.success else np.ones(10)*budget/10
    state = 100.0; rows=[]
    for y, spend in zip(years, u):
        state = (1 - depreciation) * state + 0.018 * spend
        rows.append({'Năm': y, 'Đầu tư tối ưu': spend, 'Chỉ số năng lực số': state, 'GDP gain proxy': 0.55*np.log1p(state)+0.00022*spend})
    return pd.DataFrame(rows)

# ---------- Bài 9 ----------
def labor_ai_impact(sectors: pd.DataFrame, budget: float) -> pd.DataFrame:
    df = sectors.head(8).copy()
    invest = budget * (df['ai_readiness_0_100'] / df['ai_readiness_0_100'].sum())
    displaced = df['labor_million'] * df['automation_risk_pct'] / 100 * (0.18 + invest / (budget + 1e-12) * 0.25)
    created = invest / budget * 1.25 + df['digital_index_0_100'] / 100 * 0.06
    df['AI_invest'] = invest
    df['Jobs_displaced_million'] = displaced
    df['Jobs_created_million'] = created
    df['NetJob_million'] = created - displaced
    return df[['sector_name_vi','AI_invest','Jobs_displaced_million','Jobs_created_million','NetJob_million']]

# ---------- Bài 10 ----------
def stochastic_program(total_budget: float, probs: tuple[float, float, float]) -> tuple[pd.DataFrame, float, float]:
    scenarios = ['Lạc quan', 'Cơ sở', 'Bất lợi']
    multipliers = np.array([1.18, 1.00, 0.78])
    p = np.array(probs, dtype=float); p = p / p.sum()
    base = np.array([0.85, 1.20, 0.95, 1.35])
    exp_coef = (p @ multipliers) * base
    # Expected value solution
    res = linprog(-exp_coef, A_ub=[[1,1,1,1], [0.35,-0.65,0.35,-0.65]], b_ub=[total_budget, 0], bounds=[(15,None),(10,None),(15,None),(8,None)], method='highs')
    x = res.x if res.success else np.array([15,10,15,total_budget-40])
    expected = float(x @ exp_coef)
    # perfect information proxy: solve each scenario then expectation
    pi_vals = []
    for m in multipliers:
        r = linprog(-(base*m), A_ub=[[1,1,1,1], [0.35,-0.65,0.35,-0.65]], b_ub=[total_budget,0], bounds=[(15,None),(10,None),(15,None),(8,None)], method='highs')
        pi_vals.append(float(-r.fun) if r.success else expected)
    evpi = float(p @ np.array(pi_vals) - expected)
    vss = max(0.0, evpi * 0.35)
    df = pd.DataFrame({'Hạng mục':['Hạ tầng','AI','Nhân lực','R&D'], 'Phân bổ first-stage': x, 'Hệ số kỳ vọng': exp_coef})
    return df, vss, evpi

# ---------- Bài 11 ----------
def q_learning(episodes: int, alpha: float, gamma: float, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_states, n_actions = 81, 5
    Q = np.zeros((n_states, n_actions))
    rewards = []
    for ep in range(int(episodes)):
        s = int(rng.integers(0, n_states)); total = 0.0
        for _ in range(24):
            if rng.random() < max(0.05, 0.7 * (1 - ep / max(episodes, 1))):
                a = int(rng.integers(0, n_actions))
            else:
                a = int(np.argmax(Q[s]))
            gdp_state = s // 27; d_state = (s // 9) % 3; risk_state = s % 3
            reward = [0.2, 0.6, 0.85, 0.75, 0.65][a] + 0.22*gdp_state + 0.18*d_state - 0.30*risk_state + rng.normal(0, 0.08)
            ns = int(np.clip(s + rng.integers(-6, 7) + a - 2, 0, n_states-1))
            Q[s,a] += alpha * (reward + gamma * np.max(Q[ns]) - Q[s,a])
            s = ns; total += reward
        rewards.append(total)
    curve = pd.DataFrame({'Episode': np.arange(1, int(episodes)+1), 'Reward': rewards})
    curve['Reward_smooth'] = curve['Reward'].rolling(max(5, int(episodes)//40), min_periods=1).mean()
    pol = pd.DataFrame({'State': np.arange(n_states), 'Best_Action': np.argmax(Q, axis=1)})
    return curve, pol

# ---------- Bài 12 ----------
def scenarios_summary() -> pd.DataFrame:
    data = [
        ['S1 Truyền thống', .54, .42, .36, .48, .40],
        ['S2 Cân bằng', .68, .61, .58, .64, .57],
        ['S3 Số hóa nhanh', .77, .56, .60, .59, .62],
        ['S4 AI dẫn dắt', .86, .50, .55, .52, .70],
        ['S5 Bao trùm + an toàn', .78, .75, .70, .73, .69],
    ]
    return pd.DataFrame(data, columns=['Kịch bản','Tăng trưởng','Bao trùm','Môi trường','An ninh','AI Readiness'])
