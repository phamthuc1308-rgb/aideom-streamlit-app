# AIDEOM-VN Streamlit Dashboard

Ứng dụng Streamlit cho bộ bài tập **Mô hình ra quyết định – Phát triển kinh tế Việt Nam trong kỉ nguyên AI**.

## Chạy local

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Nên dùng Python 3.10 hoặc 3.11 để tránh lỗi cài các thư viện tối ưu hóa.

## Deploy Streamlit Cloud

1. Tạo repository GitHub mới, ví dụ `AIDEOM-VN`.
2. Upload toàn bộ thư mục này lên GitHub.
3. Vào https://share.streamlit.io hoặc Streamlit Community Cloud.
4. Chọn repo, branch `main`, file chính `app.py`.
5. Bấm **Deploy**.

## Cấu trúc

```text
AIDEOM-VN/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
├── utils/
└── assets/style.css
```
