import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from part2.data_pipeline import DataPipeline, load_and_preprocess_raw_data
from part1.ols_implementation import ols_fit, vif
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv

def calc_metrics(y_true, y_pred):
    """Tính các chỉ số đánh giá MAE, RMSE, R_squared"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    tss = np.sum((y_true - np.mean(y_true))**2)
    rss = np.sum((y_true - y_pred)**2)
    r2 = 1 - (rss / tss)
    
    return mae, rmse, r2

def run_model_comparison():
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "part2", "data")
    data_path = os.path.join(data_dir, "AirQualityUCI.csv")
    
    print("[*] Đang load và tiền xử lý data AirQualityUCI...")
    X_raw, y_raw = load_and_preprocess_raw_data(data_path)
    
    if X_raw is None:
        print("Không thể load được dữ liệu. Bạn hãy đảm bảo file CSV tồn tại!")
        return
        
    # 2. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    # 3. Pipeline Transform
    pipeline = DataPipeline(target_col='CO(GT)')
    X_train_tf = pipeline.fit_transform(X_train)
    X_test_tf = pipeline.transform(X_test)
    
    print(f"Kích thước X_train sau pipeline: {X_train_tf.shape}")
    print(f"Kích thước X_test sau pipeline: {X_test_tf.shape}")
    
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    # ---------------------------------------------------------
    # Mô hình 1: OLS Cơ Bản
    # ---------------------------------------------------------
    print("\n--- Đang huấn luyện Mô hình 1: OLS Cơ bản ---")
    beta_ols, _ = ols_fit(X_train_tf, y_train)
    y_pred_ols = X_test_tf @ beta_ols
    mae_ols, rmse_ols, r2_ols = calc_metrics(y_test, y_pred_ols)

    # ---------------------------------------------------------
    # Mô hình 2: OLS Chọn Biến (Dựa trên VIF)
    # ---------------------------------------------------------
    print("\n--- Đang huấn luyện Mô hình 2: OLS Chọn Biến (VIF) ---")
    vif_scores = vif(X_train_tf) # VIF tự động bỏ cột intercept 0 để tính
    
    # Giữ lại cột 0 (Intercept) và các cột có VIF <= 10
    cols_to_keep = [0]
    for j, v in enumerate(vif_scores):
        if v <= 10:
            cols_to_keep.append(j + 1)
            
    num_dropped = X_train_tf.shape[1] - len(cols_to_keep)
    print(f"[*] Số biến độc lập bị loại bỏ do đa cộng tuyến (VIF > 10): {num_dropped}")
    
    X_train_sel = X_train_tf[:, cols_to_keep]
    X_test_sel = X_test_tf[:, cols_to_keep]
    
    beta_ols_sel, _ = ols_fit(X_train_sel, y_train)
    y_pred_sel = X_test_sel @ beta_ols_sel
    mae_sel, rmse_sel, r2_sel = calc_metrics(y_test, y_pred_sel)

    # ---------------------------------------------------------
    # Mô hình 3: Ridge Regression (Tối ưu Lambda qua CV)
    # ---------------------------------------------------------
    print("\n--- Đang huấn luyện Mô hình 3: Ridge Regression ---")
    lambdas = np.logspace(-3, 3, 50)
    best_lam = None
    best_mse = float('inf')
    
    print("[*] Đang thực thi K-Fold Cross Validation (k=5) tìm Lambda tốt nhất...")
    for lam in lambdas:
        # Gọi kfold_cv tự build
        mse_cv = kfold_cv(X_train_tf, y_train, k=5, model_func=ridge_fit, lam=lam)
        if mse_cv < best_mse:
            best_mse = mse_cv
            best_lam = lam
            
    print(f"[*] Lambda tối ưu tìm được: {best_lam:.4f}")
    
    beta_ridge = ridge_fit(X_train_tf, y_train, lam=best_lam)
    y_pred_ridge = X_test_tf @ beta_ridge
    mae_ridge, rmse_ridge, r2_ridge = calc_metrics(y_test, y_pred_ridge)

    # ---------------------------------------------------------
    # BẢNG TỔNG HỢP SO SÁNH
    # ---------------------------------------------------------
    results = pd.DataFrame({
        'Model': ['Mô hình 1 (OLS Cơ bản)', 'Mô hình 2 (OLS Chọn biến VIF)', 'Mô hình 3 (Ridge Tối ưu CV)'],
        'MAE': [mae_ols, mae_sel, mae_ridge],
        'RMSE': [rmse_ols, rmse_sel, rmse_ridge],
        'R_squared': [r2_ols, r2_sel, r2_ridge]
    })
    
    print("\n================ BẢNG SO SÁNH TRÊN TẬP TEST ================")
    print(results.to_string(index=False))
    print("============================================================")

if __name__ == '__main__':
    run_model_comparison()
