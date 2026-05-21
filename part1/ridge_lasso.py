import numpy as np
import matplotlib.pyplot as plt
from part1.ols_implementation import _add_intercept

def ridge_fit(X, y, lam):
    """
    Cài đặt Ridge Regression bằng Đại số tuyến tính.
    Công thức: beta_hat_ridge = (X^T * X + lam * I_modified)^-1 * X^T * y
    """
    X = _add_intercept(X)
    y = np.asarray(y).reshape(-1, 1)
    
    p_plus_1 = X.shape[1]
    
    # I_modified: Ma trận đơn vị nhưng giá trị tại [0,0] bằng 0 (Không phạt intercept)
    I_mod = np.eye(p_plus_1)
    I_mod[0, 0] = 0
    
    XTX = X.T @ X
    
    try:
        matrix_to_inv = XTX + lam * I_mod
        matrix_inv = np.linalg.inv(matrix_to_inv)
    except np.linalg.LinAlgError:
        raise ValueError("Lỗi suy biến: Ma trận không thể nghịch đảo.")
        
    beta_hat_ridge = matrix_inv @ X.T @ y
    
    return beta_hat_ridge

def plot_ridge_trace(X, y, lambdas):
    """
    Lặp qua list lambdas, tính beta_hat và vẽ đồ thị sự thay đổi của các hệ số.
    """
    coefs = []
    for lam in lambdas:
        beta = ridge_fit(X, y, lam)
        # Bỏ qua hệ số chặn beta_0, chỉ lấy các hệ số của features
        coefs.append(beta[1:].flatten())
        
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    p = coefs.shape[1]
    for j in range(p):
        plt.plot(lambdas, coefs[:, j], label=f'Feature {j+1}')
        
    plt.xscale('log') # Hiển thị thang log để dễ nhìn sự hội tụ
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Coefficients (Hệ số)')
    plt.title('Ridge Trace (Đồ thị thay đổi hệ số theo Lambda)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    X_test = np.random.randn(100, 3)
    y_test = X_test @ np.array([[1.5], [-2.0], [0.5]]) + np.random.randn(100, 1)
    
    # In thử 1 giá trị lambda
    b = ridge_fit(X_test, y_test, lam=1.0)
    print("Beta_hat cho Ridge với lambda = 1.0:\n", b)
