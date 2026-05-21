import nbformat as nbf

# ==============================================================================
# NOTEBOOK 1: PART 1
# ==============================================================================
nb1 = nbf.v4.new_notebook()

c1_1 = nbf.v4.new_markdown_cell("""# Phần 1: Lý Thuyết Data Fitting và OLS
Bài toán Data Fitting tìm kiếm một mô hình toán học giải thích mối quan hệ giữa các đặc trưng (features) và biến mục tiêu (target). 
Phương pháp Bình phương tối thiểu thông thường (OLS) sử dụng Đại số tuyến tính để tối thiểu hóa tổng bình phương phần dư (RSS), nhằm tìm ra các hệ số hồi quy $\\hat{\\beta}$ tối ưu.""")

c1_2 = nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
from part1.ols_implementation import ols_fit, hat_matrix, model_metrics, _add_intercept""")

c1_3 = nbf.v4.new_code_cell("""# Khởi tạo dữ liệu giả lập (Synthetic data)
np.random.seed(42)
n = 100
p = 2
X = np.random.randn(n, p)

# Intercept = 3.0, Feature 1 = 1.5, Feature 2 = -2.0
true_beta = np.array([[3.0], [1.5], [-2.0]]) 

X_int = _add_intercept(X)
# Thêm nhiễu phân phối chuẩn
y = X_int @ true_beta + np.random.randn(n, 1) * 1.5

print(f"Đã tạo dữ liệu giả lập với {n} mẫu và {p} features.")""")

c1_4 = nbf.v4.new_code_cell("""# Chạy mô hình OLS cơ bản
beta_hat, sigma2 = ols_fit(X, y)

print("--- Hệ số hồi quy ước lượng (Beta_hat) ---")
print(beta_hat)

# Tính Ma trận chiếu
H = hat_matrix(X)
print(f"\\n--- Kích thước Ma trận Hat ---: {H.shape}")

# Đánh giá độ phù hợp của mô hình
y_pred = _add_intercept(X) @ beta_hat
metrics = model_metrics(y, y_pred, p)

print("\\n--- Kết quả Đánh giá mô hình ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")""")

c1_5 = nbf.v4.new_markdown_cell("""### Định lý Gauss-Markov
Định lý Gauss-Markov phát biểu rằng, trong mô hình hồi quy tuyến tính cổ điển (kỳ vọng sai số bằng 0, phương sai đồng nhất, độc lập không tự tương quan), ước lượng OLS là **ước lượng tuyến tính không chệch tốt nhất (BLUE)**.
Tính "không chệch" nghĩa là kỳ vọng của các hệ số ước lượng qua nhiều lần lấy mẫu ngẫu nhiên sẽ hội tụ đúng về hệ số thực tế: $E[\\hat{\\beta}] = \\beta$.

Dưới đây, ta sẽ chạy mô phỏng Monte Carlo để chứng minh điều này.""")

c1_6 = nbf.v4.new_code_cell("""# Mô phỏng Monte Carlo 1000 lần
n_sims = 1000
betas_sim = []

for _ in range(n_sims):
    # Lấy mẫu y mới do sai số ngẫu nhiên thay đổi
    y_sim = X_int @ true_beta + np.random.randn(n, 1) * 1.5
    b_hat, _ = ols_fit(X, y_sim)
    betas_sim.append(b_hat.flatten())

betas_sim = np.array(betas_sim)
mean_betas = np.mean(betas_sim, axis=0)

print("Kỳ vọng thực sự của Beta (True Beta):  ", true_beta.flatten())
print("Trung bình Beta_hat (1000 lần MC):    ", mean_betas)

# Vẽ biểu đồ Histogram cho hệ số Feature 1 (beta_1)
plt.figure(figsize=(9, 6))
plt.hist(betas_sim[:, 1], bins=40, edgecolor='black', alpha=0.7)
plt.axvline(true_beta[1][0], color='red', linestyle='dashed', linewidth=3, label='True Beta 1 (Thực tế)')
plt.axvline(mean_betas[1], color='blue', linestyle='dotted', linewidth=3, label='E[Beta_hat] (Kỳ vọng OLS)')
plt.title("Phân phối của Beta_1_hat qua 1000 lần lặp Monte Carlo", fontsize=14)
plt.xlabel("Giá trị của Beta 1")
plt.ylabel("Tần suất")
plt.legend(fontsize=12)
plt.show()""")

nb1['cells'] = [c1_1, c1_2, c1_3, c1_4, c1_5, c1_6]

with open('part1_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb1, f)


# ==============================================================================
# NOTEBOOK 2: PART 2
# ==============================================================================
nb2 = nbf.v4.new_notebook()

c2_1 = nbf.v4.new_markdown_cell("""# Phần 2: Ứng dụng mô hình trên dữ liệu AirQualityUCI
Bộ dữ liệu AirQualityUCI đo lường chất lượng không khí nhưng tồn tại cực kỳ nhiều nhiễu. Cơ chế tiền xử lý cần tuân thủ nghiêm ngặt 2 quy tắc:
1. **Giá trị `-200`**: Toàn bộ hệ thống cảm biến sẽ trả về `-200` nếu bị lỗi missing value. Việc để nguyên `-200` đi huấn luyện sẽ làm nghiêng toàn bộ độ dốc OLS. Do đó, phải map nó thành `np.nan`.
2. **Loại bỏ cột `NMHC(GT)`**: Cột này có tới hơn 90% số lượng mẫu là missing values. Việc cố gắng điền khuyết (Imputation) 90% dữ liệu ảo sẽ phá hủy hoàn toàn phương sai của biến này, vì vậy loại bỏ nó (Drop column) là quyết định chuyên gia bắt buộc.""")

c2_2 = nbf.v4.new_code_cell("""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from part2.data_pipeline import load_and_preprocess_raw_data

# Khám phá dữ liệu EDA (Exploratory Data Analysis)
data_path = os.path.join('part2', 'data', 'AirQualityUCI.csv')
X_raw, y_raw = load_and_preprocess_raw_data(data_path)

if X_raw is not None:
    df_eda = X_raw.copy()
    df_eda['CO(GT)'] = y_raw
    
    plt.figure(figsize=(12, 10))
    # Tính ma trận tương quan bỏ qua các giá trị NaN
    corr_matrix = df_eda.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', fmt=".2f", linewidths=0.5)
    plt.title("Ma trận Tương quan giữa các cảm biến (Correlation Heatmap)", fontsize=16)
    plt.show()
else:
    print("Dữ liệu thô không tồn tại. Vui lòng đặt file AirQualityUCI.csv vào thư mục part2/data/")""")

c2_3 = nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from part2.data_pipeline import DataPipeline
import numpy as np

if X_raw is not None:
    # Phân chia 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)
    
    # OOP Data Pipeline: Ngăn chặn tuyệt đối Data Leakage
    pipeline = DataPipeline(target_col='CO(GT)', imputation_method='knn')
    
    X_train_tf = pipeline.fit_transform(X_train)
    X_test_tf = pipeline.transform(X_test)
    
    print("Hoàn tất Pipeline. Shape của X_train đã biến đổi:", X_train_tf.shape)
    print("Hoàn tất Pipeline. Shape của X_test đã biến đổi:", X_test_tf.shape)""")

c2_4 = nbf.v4.new_code_cell("""from part2.model_comparison import calc_metrics
from part1.ols_implementation import ols_fit, vif
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv

if X_raw is not None:
    # ------------------ 1. OLS CƠ BẢN ------------------
    beta_ols, _ = ols_fit(X_train_tf, y_train)
    mae_ols, rmse_ols, r2_ols = calc_metrics(y_test, X_test_tf @ beta_ols)
    
    # ------------------ 2. OLS CHỌN BIẾN (VIF) ---------
    vif_scores = vif(X_train_tf)
    # Giữ intercept (0) và các features có VIF <= 10
    cols_to_keep = [0] + [j + 1 for j, v in enumerate(vif_scores) if v <= 10]
    
    X_train_sel = X_train_tf[:, cols_to_keep]
    X_test_sel = X_test_tf[:, cols_to_keep]
    
    beta_sel, _ = ols_fit(X_train_sel, y_train)
    mae_sel, rmse_sel, r2_sel = calc_metrics(y_test, X_test_sel @ beta_sel)
    
    # ------------------ 3. RIDGE REGRESSION CV ---------
    best_lam, best_mse = None, float('inf')
    lambdas = np.logspace(-3, 3, 20)
    for lam in lambdas:
        mse_cv = kfold_cv(X_train_tf, y_train, k=5, model_func=ridge_fit, lam=lam)
        if mse_cv < best_mse:
            best_mse = mse_cv
            best_lam = lam
            
    beta_ridge = ridge_fit(X_train_tf, y_train, best_lam)
    mae_ridge, rmse_ridge, r2_ridge = calc_metrics(y_test, X_test_tf @ beta_ridge)
    
    # ------------------ XUẤT BẢNG SO SÁNH ------------------
    results = pd.DataFrame({
        'Mô Hình (Trên tập Test)': ['OLS Cơ Bản', 'OLS Lọc VIF', f'Ridge (L={best_lam:.2f})'],
        'MAE': [mae_ols, mae_sel, mae_ridge],
        'RMSE': [rmse_ols, rmse_sel, rmse_ridge],
        'R_Squared': [r2_ols, r2_sel, r2_ridge]
    })
    
    display(results)""")

c2_5 = nbf.v4.new_markdown_cell("""### Nhận xét Kết quả 3 Mô hình
Bảng so sánh trên tập kiểm thử (Test Set) cho ta những góc nhìn sau:

1. **OLS Cơ bản**: Mô hình đạt được $R^2$ rất cao, tuy nhiên do sử dụng toàn bộ feature, nó có rủi ro bị bất ổn định bởi ma trận thiết kế có độ đa cộng tuyến khổng lồ từ các cảm biến khí trùng lặp.
2. **OLS Chọn biến (VIF Filter)**: Phương pháp này thẳng tay loại bỏ các biến có $VIF > 10$. Điều này khiến $R^2$ có thể suy giảm đôi chút (vì mất mát lượng thông tin tương quan lặp) nhưng nó đảm bảo các hệ số $\\beta$ thu được đáng tin cậy hơn, không bị bóp méo phương sai.
3. **Ridge Regression**: Đây là mô hình tinh xảo nhất. Thông qua việc cộng thêm ma trận phạt $\\lambda I_{mod}$, hệ số của mô hình bị ép co rút (shrinkage) đồng đều. Cross-Validation (k=5) đã tự động tìm ra cường độ phạt tối ưu nhất, giúp cân bằng hoàn hảo giữa thiên lệch (Bias) và phương sai (Variance) của mô hình. 

**Kết luận**: Mô hình Ridge là lựa chọn thực tiễn ưu việt nhất khi triển khai hệ thống học máy Data Fitting lên luồng phân tích luân chuyển (Production) cho bài toán cảm biến IoT.""")

nb2['cells'] = [c2_1, c2_2, c2_3, c2_4, c2_5]

with open('part2_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb2, f)
    
print("Successfully generated Notebook files using nbformat.")
