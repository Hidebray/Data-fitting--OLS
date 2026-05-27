import numpy as np
import scipy.stats


def _add_intercept(X):
    """
    @brief  Thêm cột hằng số 1 (intercept) vào đầu ma trận X nếu chưa có.

    @input  X : array-like, shape (n,) hoặc (n, p)
                Ma trận đặc trưng đầu vào.

    @output X_with_intercept : np.ndarray, shape (n, p+1)
                Ma trận X đã có cột đầu là toàn số 1.
                Nếu cột đầu đã là toàn số 1 thì trả về X nguyên vẹn.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Kiểm tra xem cột đầu tiên có phải toàn số 1 không
    if X.shape[1] > 0 and np.allclose(X[:, 0], 1):
        return X

    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def ols_fit(X, y):
    """
    @brief  Ước lượng hệ số hồi quy bằng phương pháp Ordinary Least Squares (OLS).
            Công thức: beta_hat = (X^T X)^{-1} X^T y.
            Ước lượng phương sai nhiễu: sigma^2 = RSS / (n - p - 1).

    @input  X         : array-like, shape (n, p)
                        Ma trận đặc trưng (chưa có cột intercept).
            y         : array-like, shape (n,) hoặc (n, 1)
                        Vector biến mục tiêu.

    @output beta_hat  : np.ndarray, shape (p+1, 1)
                        Vector hệ số hồi quy ước lượng (gồm intercept ở vị trí 0).
            sigma_squared : float
                        Ước lượng không chệch của phương sai nhiễu sigma^2.

    @raises ValueError  Nếu ma trận X^T X suy biến (không khả nghịch).
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

    df = n - p - 1
    if df <= 0:
        raise ValueError(
        "Số bậc tự do không hợp lệ: n phải lớn hơn p + 1."
    )

    sigma_squared = rss / df

    return beta_hat, sigma_squared


def hat_matrix(X):
    """
    @brief  Tính ma trận Hat (ma trận chiếu / projection matrix).
            Công thức: H = X (X^T X)^{-1} X^T.
            Tính chất: H^2 = H (idempotent), H^T = H (đối xứng),
            giá trị riêng chỉ là 0 hoặc 1, rank(H) = p + 1.

    @input  X : array-like, shape (n, p)
                Ma trận đặc trưng (chưa có cột intercept).

    @output H : np.ndarray, shape (n, n)
                Ma trận Hat.
    """
    X = _add_intercept(X)
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    H = X @ XTX_inv @ X.T
    return H


def model_metrics(y, y_hat, p):
    """
    @brief  Tính các chỉ số đánh giá mô hình hồi quy tuyến tính:
            RSS, TSS, R^2, R^2 hiệu chỉnh và thống kê F.

    @input  y     : array-like, shape (n,) hoặc (n, 1)
                    Giá trị thực tế của biến mục tiêu.
            y_hat : array-like, shape (n,) hoặc (n, 1)
                    Giá trị dự đoán của mô hình.
            p     : int
                    Số biến đặc trưng (không tính intercept).

    @output metrics : dict với các khóa:
                - "RSS"                : Residual Sum of Squares.
                - "TSS"                : Total Sum of Squares.
                - "R_squared"          : Hệ số xác định R^2.
                - "Adjusted_R_squared" : R^2 hiệu chỉnh.
                - "F_statistic"        : Thống kê F cho kiểm định tổng thể.
    """
    y = np.asarray(y).reshape(-1, 1)
    y_hat = np.asarray(y_hat).reshape(-1, 1)
    n = y.shape[0]

    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    r_squared = 1 - (rss / tss)
    adj_r_squared = 1 - ((n - 1) / (n - p - 1) * (1 - r_squared))

    if p == 0:
        f_statistic = np.nan
    else:
        f_statistic = ((tss - rss) / p) / (rss / (n - p - 1))

    return {
        "RSS": rss,
        "TSS": tss,
        "R_squared": r_squared,
        "Adjusted_R_squared": adj_r_squared,
        "F_statistic": f_statistic,
    }


def coef_inference(X, y, beta_hat, sigma2):
    """
    @brief  Tính các thống kê suy luận cho vector hệ số hồi quy:
            standard errors, t-statistics, p-values (kiểm định 2 phía H0: beta_j = 0)
            và khoảng tin cậy 95%.

    @input  X        : array-like, shape (n, p)
                       Ma trận đặc trưng (chưa có cột intercept).
            y        : array-like, shape (n,) hoặc (n, 1)
                       Vector biến mục tiêu (chỉ dùng để xác định n).
            beta_hat : np.ndarray, shape (p+1, 1)
                       Vector hệ số ước lượng (kết quả của ols_fit).
            sigma2   : float
                       Ước lượng phương sai nhiễu sigma^2 (kết quả của ols_fit).

    @output se     : np.ndarray, shape (p+1, 1)
                     Standard errors của từng hệ số.
            t_stat : np.ndarray, shape (p+1, 1)
                     Giá trị thống kê t cho từng hệ số.
            p_value: np.ndarray, shape (p+1, 1)
                     P-value (kiểm định 2 phía) cho từng hệ số.
            ci_95  : np.ndarray, shape (p+1, 2)
                     Khoảng tin cậy 95%: cột 0 là cận dưới, cột 1 là cận trên.
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
    @brief  Tính Hệ số Phóng Đại Phương Sai (Variance Inflation Factor) cho từng
            biến độc lập nhằm phát hiện đa cộng tuyến.
            Công thức: VIF_j = 1 / (1 - R^2_j), trong đó R^2_j là hệ số xác định
            khi hồi quy biến X_j theo các biến còn lại.
            VIF > 10 cho thấy đa cộng tuyến nghiêm trọng.

    @input  X    : array-like, shape (n, p) hoặc (n, p+1)
                   Ma trận đặc trưng (có thể có hoặc không có cột intercept).

    @output vifs : np.ndarray, shape (p,)
                   Mảng giá trị VIF cho từng biến đặc trưng (không tính intercept).
                   Trả về np.inf nếu biến đó hoàn toàn tuyến tính phụ thuộc vào
                   các biến còn lại (R^2_j = 1).
    """
    X_features = np.asarray(X)

    # VIF chỉ tính cho các biến độc lập, loại bỏ intercept nếu có
    if X_features.shape[1] > 0 and np.allclose(X_features[:, 0], 1):
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
