import os
import pathlib
import json

def create_project_structure():
    # Thư mục gốc chứa file script
    base_dir = pathlib.Path(__file__).parent.resolve()
    
    # 1. Định nghĩa các thư mục cần tạo
    directories = [
        "report",
        "part1",
        "part2",
        "part2/data"
    ]
    
    # 2. Định nghĩa cấu trúc notebook rỗng hợp lệ
    empty_notebook = json.dumps({
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }, indent=1)

    # 3. Định nghĩa các file cần tạo và nội dung mặc định của chúng
    files = {
        "report/report.tex": "",
        "part1/__init__.py": "",
        "part1/ols_implementation.py": "",
        "part1/ridge_lasso.py": "",
        "part1/cross_validation.py": "",
        "part1/residual_analysis.py": "",
        "part2/__init__.py": "",
        "part2/data_pipeline.py": "",
        "part2/model_comparison.py": "",
        "part2/advanced_methods.py": "",
        "part1_notebook.ipynb": empty_notebook,
        "part2_notebook.ipynb": empty_notebook
    }

    # Nội dung cho requirements.txt
    requirements_content = """numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0
notebook>=7.0.0
"""

    # Nội dung template cho README.md
    readme_content = """# Data Fitting Project

## Hướng dẫn cài đặt môi trường

Dự án này yêu cầu một số thư viện Python cơ bản cho Khoa học dữ liệu (Data Science). Khuyến nghị sử dụng môi trường ảo (virtual environment) để tránh xung đột thư viện.

1. **Tạo môi trường ảo (tùy chọn nhưng khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Kích hoạt trên Windows:
   venv\\Scripts\\activate
   
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
"""

    files["requirements.txt"] = requirements_content
    files["README.md"] = readme_content

    print("Đang tiến hành khởi tạo cấu trúc dự án Data Fitting...\\n")

    # Tạo các thư mục
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  [+] Đã tạo thư mục: {dir_path}")

    print("\\n--------------------------------------------------\\n")

    # Tạo các file
    for file_path, content in files.items():
        path = base_dir / file_path
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [+] Đã tạo file: {path}")

    print("\\n✅ Khởi tạo dự án thành công tại New folder! Bạn có thể bắt đầu code ngay bây giờ.")

if __name__ == "__main__":
    create_project_structure()
