import numpy as np

class BayesianLinearRegression:
    def __init__(self, m0, S0, sigma2):
        """
        @brief Khởi tạo mô hình Bayesian Linear Regression.
        @input m0 (numpy array): Vector kỳ vọng tiên nghiệm (Prior Mean) [p x 1].
        @input S0 (numpy array): Ma trận hiệp phương sai tiên nghiệm (Prior Covariance) [p x p].
        @input sigma2 (float): Phương sai nhiễu đã biết (Noise variance).
        """
        self.m0 = np.asarray(m0).reshape(-1, 1)
        self.S0 = np.asarray(S0)
        self.sigma2 = sigma2
        
        self.m_n = None
        self.S_n = None
        
    def fit(self, X, y):
        """
        @brief Huấn luyện mô hình Bayesian, cập nhật Prior thành Posterior.
        @input X (numpy array): Ma trận đặc trưng [n x p].
        @input y (numpy array): Vector mục tiêu [n x 1].
        @output self: Trả về chính object chứa m_n và S_n đã cập nhật.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        
        S0_inv = np.linalg.inv(self.S0)
        XTX = X.T @ X
        XTy = X.T @ y
        
        self.S_n = np.linalg.inv(S0_inv + XTX / self.sigma2)
        self.m_n = self.S_n @ (S0_inv @ self.m0 + XTy / self.sigma2)
        
        return self
        
    def predict(self, X):
        """
        @brief Dự đoán giá trị y và tính phương sai dự đoán.
        @input X (numpy array): Ma trận đặc trưng tập kiểm thử [m x p].
        @output tuple: (y_pred, var_pred) - Giá trị dự đoán và phương sai dự đoán [m x 1].
        """
        X = np.asarray(X)
        y_pred = X @ self.m_n
        var_pred = self.sigma2 + np.sum((X @ self.S_n) * X, axis=1).reshape(-1, 1)
        
        return y_pred, var_pred