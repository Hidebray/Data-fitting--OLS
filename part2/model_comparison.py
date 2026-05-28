import numpy as np
import pandas as pd

from part1.ols_implementation import ols_fit, vif
from part1.ridge_lasso import ridge_fit
from part1.cross_validation import kfold_cv

def calc_metrics(y_true, y_pred):
    """
    @brief Tính các chỉ số đánh giá MAE, RMSE, R_squared.
    @input y_true (numpy array): Vector giá trị thực tế.
    @input y_pred (numpy array): Vector giá trị dự đoán.
    @output tuple: (mae, rmse, r2).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    tss = np.sum((y_true - np.mean(y_true))**2)
    rss = np.sum((y_true - y_pred)**2)
    r2 = 1 - (rss / tss)
    
    return mae, rmse, r2

def evaluate_models(X_train_tf, X_test_tf, y_train, y_test):
    """
    @brief Chạy và so sánh 3 mô hình: OLS, OLS (chọn biến qua VIF), và Ridge.
    @input X_train_tf (numpy array): Dữ liệu huấn luyện đã qua pipeline.
    @input X_test_tf (numpy array): Dữ liệu kiểm thử đã qua pipeline.
    @input y_train (numpy array): Target huấn luyện.
    @input y_test (numpy array): Target kiểm thử.
    @output dict: Chứa bảng so sánh (metrics_table), weights của Ridge, và dữ liệu phần dư.
    """
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    # 1. Mô hình OLS Cơ Bản
    beta_ols, _ = ols_fit(X_train_tf, y_train)
    y_pred_ols = X_test_tf @ beta_ols
    mae_ols, rmse_ols, r2_ols = calc_metrics(y_test, y_pred_ols)

    # 2. Mô hình OLS Chọn Biến (Dựa trên VIF)
    vif_scores = vif(X_train_tf)
    cols_to_keep = [0]
    for j, v in enumerate(vif_scores):
        if v <= 10:
            cols_to_keep.append(j + 1)
            
    X_train_sel = X_train_tf[:, cols_to_keep]
    X_test_sel = X_test_tf[:, cols_to_keep]
    
    beta_ols_sel, _ = ols_fit(X_train_sel, y_train)
    y_pred_sel = X_test_sel @ beta_ols_sel
    mae_sel, rmse_sel, r2_sel = calc_metrics(y_test, y_pred_sel)

    # 3. Mô hình Ridge Regression (CV)
    lambdas = np.logspace(-3, 3, 50)
    best_lam = None
    best_mse = float('inf')
    
    for lam in lambdas:
        mse_cv = kfold_cv(X_train_tf, y_train, k=5, model_func=ridge_fit, lam=lam)
        if mse_cv < best_mse:
            best_mse = mse_cv
            best_lam = lam
            
    beta_ridge = ridge_fit(X_train_tf, y_train, lam=best_lam)
    y_pred_ridge = X_test_tf @ beta_ridge
    mae_ridge, rmse_ridge, r2_ridge = calc_metrics(y_test, y_pred_ridge)

    # 4. Đóng gói toàn bộ kết quả phục vụ cho Notebook vẽ biểu đồ
    results_df = pd.DataFrame({
        'Model': ['OLS Basic', 'OLS selected VIF', f'Ridge (lam={best_lam:.4f})'],
        'MAE': [mae_ols, mae_sel, mae_ridge],
        'RMSE': [rmse_ols, rmse_sel, rmse_ridge],
        'R_squared': [r2_ols, r2_sel, r2_ridge]
    })
    
    output_data = {
        "metrics_table": results_df,
        "best_ridge_weights": beta_ridge.flatten(),
        "residuals_data": {
            "y_true": y_test.flatten(),
            "y_pred_ridge": y_pred_ridge.flatten()
        }
    }
    
    return output_data