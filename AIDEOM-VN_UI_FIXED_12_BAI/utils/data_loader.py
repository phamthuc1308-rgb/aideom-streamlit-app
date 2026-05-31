"""Data loading utilities with automatic CSV creation for AIDEOM-VN."""
from __future__ import annotations
from pathlib import Path
from io import StringIO
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

MACRO_CSV = """year,GDP_trillion_VND,GDP_billion_USD,GDP_growth_pct,GDP_per_capita_USD,population_million,FDI_disbursed_billion_USD,exports_billion_USD,digital_economy_share_GDP_pct,labor_productivity_million_VND
2020,8044.4,346.6,2.91,3521,97.58,19.98,282.6,12.0,151.2
2021,8487.5,366.1,2.58,3717,98.51,19.74,336.3,12.7,171.3
2022,9513.3,408.8,8.02,4163,99.46,22.40,371.3,14.3,188.1
2023,10221.8,430.0,5.05,4347,100.30,23.18,355.5,16.5,199.3
2024,11511.9,476.3,7.09,4700,101.30,25.35,405.5,18.3,221.9
2025,12847.6,514.0,8.02,5026,102.30,27.60,475.0,19.5,245.0
"""
SECTORS_CSV = """sector_id,sector_name_vi,sector_name_en,gdp_share_2024_pct,growth_rate_2024_pct,labor_million,labor_share_pct,export_billion_USD,digital_index_0_100,ai_readiness_0_100,fdi_attraction_billion_USD,spillover_coef_0_1,automation_risk_pct,rd_intensity_pct
1,Nông-Lâm-Thủy sản,Agriculture-Forestry-Fishery,11.86,3.27,13.20,26.5,40.5,28,15,2.1,0.35,18,0.15
2,Công nghiệp chế biến chế tạo,Manufacturing,24.10,9.64,11.50,23.1,290.9,68,55,18.6,0.78,42,0.62
3,Xây dựng,Construction,7.04,7.45,4.80,9.6,2.5,35,20,0.8,0.42,25,0.18
4,Khai khoáng,Mining,3.36,-1.20,0.30,0.6,8.2,50,30,0.5,0.30,55,0.22
5,Bán buôn-bán lẻ,Wholesale-Retail,9.85,7.10,7.80,15.7,5.5,72,48,3.2,0.55,38,0.10
6,Tài chính-Ngân hàng,Finance-Banking,5.10,7.36,0.55,1.1,1.2,88,72,1.5,0.85,52,0.45
7,Logistics-Vận tải,Logistics-Transport,4.20,9.93,1.95,3.9,3.1,64,42,2.3,0.72,35,0.20
8,CNTT-Truyền thông,ICT,8.00,7.85,0.62,1.2,178.0,92,88,5.0,0.92,28,1.20
9,Giáo dục-Đào tạo,Education,4.40,6.42,2.15,4.3,0.0,55,38,0.2,0.65,22,0.30
10,Y tế,Healthcare,3.10,6.85,0.75,1.5,0.0,60,45,0.3,0.60,18,0.35
"""
REGIONS_CSV = """region_id,region_name_vi,region_name_en,population_million,grdp_trillion_VND,grdp_growth_pct,grdp_per_capita_million_VND,fdi_registered_billion_USD,exports_billion_USD,digital_index_0_100,ai_readiness_0_100,trained_labor_pct,gini_coef,rd_intensity_pct,internet_penetration_pct
1,Trung du miền núi phía Bắc,Northern Midlands and Mountains,14.2,810,8.50,57.0,3.5,42.5,38,22,21.5,0.405,0.18,72
2,Đồng bằng sông Hồng,Red River Delta,23.5,3580,7.90,152.3,20.0,132.0,78,68,36.8,0.358,0.85,92
3,Bắc Trung Bộ và duyên hải miền Trung,North Central and South Central Coast,20.8,1820,6.85,87.5,8.2,68.5,55,40,27.5,0.372,0.32,84
4,Tây Nguyên,Central Highlands,6.1,420,7.20,68.9,0.8,2.8,32,18,18.2,0.412,0.15,68
5,Đông Nam Bộ,Southeast,19.2,3050,7.50,158.9,18.5,128.5,82,75,42.5,0.385,0.78,94
6,Đồng bằng sông Cửu Long,Mekong Delta,17.6,1410,6.55,80.5,2.1,32.5,48,30,16.8,0.392,0.22,78
"""

def ensure_csv(filename: str, content: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path

@pd.api.extensions.register_dataframe_accessor("aideom")
class _Accessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

@__import__('streamlit').cache_data(show_spinner=False)
def load_macro() -> pd.DataFrame:
    p = ensure_csv("vietnam_macro_2020_2025.csv", MACRO_CSV)
    df = pd.read_csv(p)
    # Add exercise-specific columns if absent.
    if "K_accumulated_trillion_VND" not in df:
        df["K_accumulated_trillion_VND"] = [16500,17800,19600,21300,23500,25900]
    if "labor_million" not in df:
        df["labor_million"] = [53.6,50.5,51.7,52.4,52.9,53.4]
    if "ai_firms_thousand" not in df:
        df["ai_firms_thousand"] = [55.6,60.2,65.4,67.0,73.8,80.1]
    if "trained_labor_pct" not in df:
        df["trained_labor_pct"] = [24.1,26.1,26.2,27.0,28.4,29.2]
    return df.sort_values("year").reset_index(drop=True)

@__import__('streamlit').cache_data(show_spinner=False)
def load_sectors() -> pd.DataFrame:
    return pd.read_csv(ensure_csv("vietnam_sectors_2024.csv", SECTORS_CSV))

@__import__('streamlit').cache_data(show_spinner=False)
def load_regions() -> pd.DataFrame:
    return pd.read_csv(ensure_csv("vietnam_regions_2024.csv", REGIONS_CSV))
