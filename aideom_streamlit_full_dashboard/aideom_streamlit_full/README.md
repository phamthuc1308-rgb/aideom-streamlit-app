# AIDEOM-VN Streamlit Dashboard

Bản Streamlit bổ sung các phần từ website HTML vào dashboard tương tác:
- kịch bản tăng trưởng,
- kết quả số,
- biểu đồ,
- nhận xét/chính sách,
- đủ 12 bài AIDEOM-VN.

## Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud
Upload toàn bộ thư mục này lên GitHub, chọn `app.py` làm entry point.

## Cấu trúc
```text
app.py
requirements.txt
data/
  vietnam_macro_2020_2025.csv
  vietnam_regions_2024.csv
  vietnam_sectors_2024.csv
```
