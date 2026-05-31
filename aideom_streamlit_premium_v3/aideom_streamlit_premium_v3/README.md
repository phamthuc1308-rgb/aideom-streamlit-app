# AIDEOM-VN Premium Streamlit Dashboard V3

Bản V3 sửa lỗi các phần Bài 3, Bài 4, Bài 5 và đổi UI/UX theo phong cách web học thuật/premium: sidebar navy, nền giấy, tiêu đề serif, thẻ KPI, khối nhận xét chính sách và biểu đồ tương tác.

## Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud

- Repository: repo GitHub của bạn
- Branch: main
- Main file path: `app.py`

## Ghi chú sửa lỗi

- Bài 3: sửa cách xử lý tiêu chí rủi ro tự động hóa. Rủi ro được đảo chiều thành "Giảm rủi ro" rồi cộng theo trọng số, không trừ sai dấu.
- Bài 4: sửa mô hình LP công bằng vùng miền bằng `scipy.optimize.linprog`, biến M được tuyến tính hóa đúng.
- Bài 5: thay bằng bộ giải brute-force 2^15 để tránh lỗi solver PuLP trên Streamlit Cloud; vẫn giữ đầy đủ ràng buộc MIP của đề.
