# AIDEOM-VN Streamlit Dashboard

Dashboard Streamlit chuyển từ bản HTML `AIDEOM_VN_Enhanced.html`, bao phủ đủ 12 bài trong đề AIDEOM-VN.

## Cách chạy local

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc

```text
AIDEOM_VN_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   └── vietnam_regions_2024.csv
└── outputs/
```

## Deploy lên Streamlit Cloud

1. Tạo repository GitHub mới, ví dụ `AIDEOM_VN_Streamlit`.
2. Upload toàn bộ thư mục này lên GitHub.
3. Vào Streamlit Cloud → New app → chọn repo → main file: `app.py`.
4. Bấm Deploy.

## Ghi chú

- Các bài nặng như NSGA-II, tối ưu động và Q-learning được triển khai theo hướng demo tương tác để chạy ổn định trên Streamlit Cloud.
- Có thể nâng cấp từng module bằng code từ notebook hoặc bài làm chi tiết nếu cần chấm sâu phần thuật toán.
