# AIDEOM-VN Streamlit Dashboard

Dashboard 12 bài tập mô hình ra quyết định kinh tế Việt Nam trong kỷ nguyên AI.

## Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Community Cloud

1. Tạo repository GitHub.
2. Upload toàn bộ file/thư mục:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `utils/`
3. Vào Streamlit Community Cloud, chọn repo và file chính `app.py`.
4. Deploy.

## Dữ liệu

Ứng dụng tự tạo 3 CSV nếu chưa có:
- `vietnam_macro_2020_2025.csv`
- `vietnam_sectors_2024.csv`
- `vietnam_regions_2024.csv`

Bạn có thể chỉnh trực tiếp dữ liệu trong tab Tổng quan; các mô hình và biểu đồ sẽ tự chạy lại.
