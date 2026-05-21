import numpy as np

class BayesianLinearRegression:
    def __init__(self, m0, S0, sigma2):
        """
        Khởi tạo mô hình Bayesian Linear Regression.
        
        Tham số:
        - m0 (numpy array): Vector kỳ vọng tiên nghiệm (Prior Mean) [p x 1]
        - S0 (numpy array): Ma trận hiệp phương sai tiên nghiệm (Prior Covariance) [p x p]
        - sigma2 (float): Phương sai nhiễu đã biết (Noise variance)
        """
        self.m0 = np.asarray(m0).reshape(-1, 1)
        self.S0 = np.asarray(S0)
        self.sigma2 = sigma2
        
        # Các tham số của phân phối Hậu nghiệm (Posterior) sẽ được tính toán sau
        self.m_n = None
        self.S_n = None
        
    def fit(self, X, y):
        """
        Huấn luyện mô hình Bayesian. Cập nhật Prior thành Posterior thông qua dữ liệu (X, y).
        S_n = (S_0^-1 + X^T*X / sigma2)^-1
        m_n = S_n * (S_0^-1 * m0 + X^T*y / sigma2)
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        
        # Đảo ma trận S0
        S0_inv = np.linalg.inv(self.S0)
        
        # Tính X^T * X và X^T * y
        XTX = X.T @ X
        XTy = X.T @ y
        
        # Cập nhật Hiệp phương sai hậu nghiệm S_n
        self.S_n = np.linalg.inv(S0_inv + XTX / self.sigma2)
        
        # Cập nhật Trung bình hậu nghiệm m_n
        self.m_n = self.S_n @ (S0_inv @ self.m0 + XTy / self.sigma2)
        
        return self
        
    def predict(self, X):
        """
        Dự đoán giá trị y và tính phương sai dự đoán.
        y_pred = X * m_n
        var_pred = sigma2 + diag(X * S_n * X^T)
        """
        X = np.asarray(X)
        
        # 1. Dự đoán giá trị trung bình (MAP)
        y_pred = X @ self.m_n
        
        # 2. Tính phương sai dự đoán (Dự đoán độ không chắc chắn - Epistemic + Aleatoric)
        # Thay vì tính ma trận khổng lồ X @ S_n @ X^T rồi lấy đường chéo,
        # ta tính tổng theo hàng của X @ S_n * X để tiết kiệm bộ nhớ (Broadcasting tương đương diag)
        var_pred = self.sigma2 + np.sum((X @ self.S_n) * X, axis=1).reshape(-1, 1)
        
        return y_pred, var_pred

if __name__ == '__main__':
    # ==========================================
    # Script kiểm thử Bayesian Linear Regression
    # ==========================================
    np.random.seed(42)
    n = 200
    p = 4  # Số features (bao gồm cả intercept)
    
    # Tạo X với intercept
    X_train = np.hstack((np.ones((n, 1)), np.random.randn(n, p-1)))
    true_weights = np.array([[1.5], [-2.0], [0.5], [3.0]])
    noise_variance = 0.5**2
    y_train = X_train @ true_weights + np.random.randn(n, 1) * np.sqrt(noise_variance)
    
    # Khởi tạo Prior siêu rỗng (Uninformative Prior)
    m0 = np.zeros((p, 1))
    S0 = np.eye(p) * 100.0  # Phương sai prior lớn => ít phụ thuộc prior
    
    # Huấn luyện
    blr = BayesianLinearRegression(m0, S0, sigma2=noise_variance)
    blr.fit(X_train, y_train)
    
    print("--- Trọng số dự đoán Posterior Mean (m_n) ---")
    print(blr.m_n)
    print("\nSo sánh với True Weights:")
    print(true_weights)
    
    # Dự đoán trên một tập giả định mới
    X_test = np.hstack((np.ones((5, 1)), np.random.randn(5, p-1)))
    y_pred, var_pred = blr.predict(X_test)
    
    print("\n--- Kết quả Dự đoán trên 5 mẫu mới ---")
    for i in range(5):
        print(f"Mẫu {i+1} | Giá trị: {y_pred[i][0]:.4f} +/- Độ lệch chuẩn (Căn phương sai): {np.sqrt(var_pred[i][0]):.4f}")
