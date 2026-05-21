import numpy as np
import scipy.stats

def _add_intercept(X):
    """
    Hàm phụ trợ: Tự động thêm cột số 1 (intercept) vào ma trận X nếu chưa có.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Kiểm tra xem cột đầu tiên có phải toàn số 1 không
    if np.all(X[:, 0] == 1):
        return X
    
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))

def ols_fit(X, y):
    """
    Thực hiện hồi quy Ordinary Least Squares (OLS) bằng Đại số tuyến tính.
    Trả về: beta_hat (vector hệ số), sigma_squared (phương sai nhiễu)
    """
    X = _add_intercept(X)
    y = np.asarray(y).reshape(-1, 1)
    
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    XTX = X.T @ X
    try:
        XTX_inv = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        raise ValueError("Lỗi suy biến: Ma trận X^T X bị suy biến, không thể nghịch đảo.")
        
    beta_hat = XTX_inv @ X.T @ y
    
    y_hat = X @ beta_hat
    rss = np.sum((y - y_hat) ** 2)
    
    sigma_squared = rss / (n - p - 1)
    
    return beta_hat, sigma_squared

def hat_matrix(X):
    """
    Tính ma trận Hat (Ma trận chiếu).
    """
    X = _add_intercept(X)
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    H = X @ XTX_inv @ X.T
    return H

def model_metrics(y, y_hat, p):
    """
    Tính toán các chỉ số đánh giá mô hình hồi quy.
    """
    y = np.asarray(y).reshape(-1, 1)
    y_hat = np.asarray(y_hat).reshape(-1, 1)
    n = y.shape[0]
    
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    
    r_squared = 1 - (rss / tss)
    adj_r_squared = 1 - ((n - 1) / (n - p - 1) * (1 - r_squared))
    
    f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))
    
    return {
        "RSS": rss,
        "TSS": tss,
        "R_squared": r_squared,
        "Adjusted_R_squared": adj_r_squared,
        "F_statistic": f_statistic
    }

def coef_inference(X, y, beta_hat, sigma2):
    """
    Tính Standard Errors, t-statistics, p-values và Khoảng tin cậy 95% cho hệ số.
    """
    X = _add_intercept(X)
    n, p_plus_1 = X.shape
    df = n - p_plus_1
    
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    
    var_beta = sigma2 * XTX_inv
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)
    
    t_stat = beta_hat / se
    
    # Tính p-value (kiểm định 2 phía)
    p_value = 2 * scipy.stats.t.sf(np.abs(t_stat), df)
    
    # Khoảng tin cậy 95%
    t_crit = scipy.stats.t.isf(0.025, df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se
    ci_95 = np.hstack((ci_lower, ci_upper))
    
    return se, t_stat, p_value, ci_95

def vif(X):
    """
    Tính Hệ số phóng đại phương sai (Variance Inflation Factor) cho mỗi biến độc lập.
    """
    X_features = np.asarray(X)
    
    # VIF chỉ tính cho các biến độc lập, loại bỏ intercept nếu có
    if np.all(X_features[:, 0] == 1):
        X_features = X_features[:, 1:]
        
    n, p = X_features.shape
    vifs = []
    
    for j in range(p):
        y_j = X_features[:, j]
        X_rem = np.delete(X_features, j, axis=1)
        
        try:
            beta, _ = ols_fit(X_rem, y_j)
            
            X_rem_intercept = _add_intercept(X_rem)
            y_hat_j = X_rem_intercept @ beta
            
            tss = np.sum((y_j - np.mean(y_j)) ** 2)
            rss = np.sum((y_j.reshape(-1, 1) - y_hat_j) ** 2)
            
            r2_j = 1 - (rss / tss)
            
            # Tránh chia cho 0 nếu r2_j = 1
            if r2_j >= 1.0 or np.isclose(r2_j, 1.0):
                vifs.append(np.inf)
            else:
                vif_j = 1 / (1 - r2_j)
                vifs.append(vif_j)
        except ValueError:
            # Nếu ma trận hồi quy phụ bị suy biến
            vifs.append(np.inf)
            
    return np.array(vifs)

if __name__ == '__main__':
    import numpy.testing as npt
    print("==== Bắt đầu chạy Unit Tests cho OLS ====")
    
    # ---------------------------------------------------------
    # Test 1: Simple Linear Regression (Hồi quy tuyến tính đơn)
    # ---------------------------------------------------------
    np.random.seed(42)
    n1 = 100
    X1 = np.random.randn(n1, 1)
    true_beta1 = np.array([[2.5], [1.5]]) # Intercept=2.5, Slope=1.5
    X1_with_intercept = _add_intercept(X1)
    y1 = X1_with_intercept @ true_beta1 + np.random.randn(n1, 1) * 0.5
    
    beta_hat1, sigma2_1 = ols_fit(X1, y1)
    
    # Tính trực tiếp để kiểm chứng
    XTX1 = X1_with_intercept.T @ X1_with_intercept
    XTy1 = X1_with_intercept.T @ y1
    expected_beta1 = np.linalg.inv(XTX1) @ XTy1
    
    npt.assert_almost_equal(beta_hat1, expected_beta1, decimal=5)
    print("✔️ Test 1 (Simple Linear Regression): PASS")
    
    # ---------------------------------------------------------
    # Test 2: Multiple Linear Regression (Hồi quy bội)
    # ---------------------------------------------------------
    p2 = 3
    X2 = np.random.randn(100, p2)
    true_beta2 = np.array([[1.0], [2.0], [-1.5], [0.5]]) # Intercept và 3 hệ số
    X2_with_intercept = _add_intercept(X2)
    y2 = X2_with_intercept @ true_beta2 + np.random.randn(100, 1) * 0.5
    
    beta_hat2, sigma2_2 = ols_fit(X2, y2)
    
    # Tính trực tiếp để kiểm chứng
    XTX2 = X2_with_intercept.T @ X2_with_intercept
    XTy2 = X2_with_intercept.T @ y2
    expected_beta2 = np.linalg.inv(XTX2) @ XTy2
    
    npt.assert_almost_equal(beta_hat2, expected_beta2, decimal=5)
    print("✔️ Test 2 (Multiple Linear Regression): PASS")
    
    # ---------------------------------------------------------
    # Test 3: Xử lý ngoại lệ với ma trận suy biến (Collinearity)
    # ---------------------------------------------------------
    X3 = np.column_stack((X1, X1)) # 2 cột hoàn toàn giống nhau (multicollinearity)
    try:
        ols_fit(X3, y1)
        print("❌ Test 3: FAIL (Đã không bắt được lỗi LinAlgError)")
    except ValueError as e:
        print(f"✔️ Test 3 (Singular Matrix Exception): PASS - Bắt được lỗi '{e}'")
        
    print("==== Hoàn tất Unit Tests ====")
