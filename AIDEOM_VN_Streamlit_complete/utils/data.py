from __future__ import annotations

from io import StringIO
from pathlib import Path
import pandas as pd

DATA_DIR = Path('.')

MACRO_CSV = '''year,GDP_trillion_VND,capital_stock_trillion_VND,labor_million,digital_economy_gdp_pct,ai_firms_thousand,trained_labor_pct,gdp_growth_pct
2020,8044.4,16500,53.6,12.0,55.6,24.1,2.91
2021,8487.5,17800,50.5,12.7,60.2,26.1,2.58
2022,9513.3,19600,51.7,14.3,65.4,26.2,8.02
2023,10221.8,21300,52.4,16.5,67.0,27.0,5.05
2024,11511.9,23500,52.9,18.3,73.8,28.4,7.09
2025,12847.6,25900,53.4,19.5,80.1,29.2,8.02
'''

SECTORS_CSV = '''sector_name_vi,growth_rate_2024_pct,productivity_million_vnd_per_worker,spillover_coef_0_1,export_billion_USD,labor_million,ai_readiness_0_100,automation_risk_pct,gdp_share_2024_pct,rd_intensity_pct,digital_index_0_100
Nông-Lâm-Thủy sản,3.27,103.4,0.35,40.5,13.20,15,18,11.86,0.12,35
CN chế biến chế tạo,9.64,241.2,0.78,290.9,11.50,55,42,24.10,0.65,62
Xây dựng,7.45,168.8,0.42,2.5,4.80,20,25,6.80,0.18,40
Khai khoáng,-1.20,1290.5,0.30,8.2,0.30,30,55,2.40,0.35,45
Bán buôn-bán lẻ,7.10,145.3,0.55,5.5,7.80,48,38,9.30,0.22,58
Tài chính-Ngân hàng,7.36,1072.4,0.85,1.2,0.55,72,52,5.70,0.90,78
Logistics-Vận tải,9.93,321.4,0.72,3.1,1.95,42,35,4.60,0.40,55
CNTT-Truyền thông,7.85,713.8,0.92,178.0,0.62,88,28,8.20,1.20,92
Giáo dục-Đào tạo,6.42,205.7,0.65,0.0,2.15,38,22,4.10,0.25,50
Y tế,6.85,437.1,0.60,0.0,0.75,45,18,3.80,0.30,54
'''

REGIONS_CSV = '''region_name_vi,grdp_per_capita_million_VND,fdi_registered_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,rd_intensity_pct,internet_penetration_pct,gini_coef,grdp_trillion_VND,export_billion_USD
Trung du miền núi phía Bắc,57.0,3.5,38,22,21.5,0.18,72,0.405,980,18
Đồng bằng sông Hồng,152.3,20.0,78,68,36.8,0.85,92,0.358,3520,160
Bắc Trung Bộ + DH Trung Bộ,87.5,8.2,55,40,27.5,0.32,84,0.372,1860,45
Tây Nguyên,68.9,0.8,32,18,18.2,0.15,68,0.412,430,4
Đông Nam Bộ,158.9,18.5,82,75,42.5,0.78,94,0.385,4200,190
Đồng bằng sông Cửu Long,80.5,2.1,48,30,16.8,0.22,78,0.392,1250,20
'''


def _ensure_csv(filename: str, content: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        path.write_text(content, encoding='utf-8')
    return path


def ensure_input_files() -> None:
    _ensure_csv('vietnam_macro_2020_2025.csv', MACRO_CSV)
    _ensure_csv('vietnam_sectors_2024.csv', SECTORS_CSV)
    _ensure_csv('vietnam_regions_2024.csv', REGIONS_CSV)


def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_input_files()
    macro = pd.read_csv('vietnam_macro_2020_2025.csv')
    sectors = pd.read_csv('vietnam_sectors_2024.csv')
    regions = pd.read_csv('vietnam_regions_2024.csv')
    return macro, sectors, regions


def reset_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(StringIO(MACRO_CSV)), pd.read_csv(StringIO(SECTORS_CSV)), pd.read_csv(StringIO(REGIONS_CSV))
