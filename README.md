# Data Fitting Project

## Hướng dẫn cài đặt môi trường

Dự án này yêu cầu một số thư viện Python cơ bản cho Khoa học dữ liệu (Data Science). Khuyến nghị sử dụng môi trường ảo (virtual environment) để tránh xung đột thư viện.

1. **Tạo môi trường ảo (tùy chọn nhưng khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn chạy code

- **Sử dụng Jupyter Notebook:**
  Để xem và chạy các bước phân tích tương tác, hãy khởi động Jupyter Notebook bằng lệnh sau trong terminal:
  ```bash
  jupyter notebook
  ```
  Sau đó, trình duyệt sẽ tự động mở lên. Bạn có thể click vào `part1_notebook.ipynb` hoặc `part2_notebook.ipynb` để chạy code.

- **Chạy Python Scripts:**
  Các thuật toán và pipeline được module hóa trong thư mục `part1` và `part2`. Bạn có thể chạy trực tiếp các script này trên terminal, ví dụ:
  ```bash
  python part1/ols_implementation.py
  ```

## Dữ liệu (Data)
**Lưu ý quan trọng:** Trước khi chạy các file trong `part2`, vui lòng tải dataset `AirQualityUCI.csv` và đặt nó vào thư mục `part2/data/`.
*Link: [AirQualityUCI.csv](https://www.kaggle.com/datasets/dakshbhalala/uci-air-quality-dataset)*
