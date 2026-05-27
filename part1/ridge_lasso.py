import numpy as np
import matplotlib.pyplot as plt

from part1.ols_implementation import _add_intercept


def ridge_fit(X, y, lam):
    """
    @brief  Ước lượng hệ số hồi quy bằng phương pháp Ridge Regression (L2 regularization).
            Công thức: beta_hat_ridge = (X^T X + lambda * I_mod)^{-1} X^T y,
            trong đó I_mod là ma trận đơn vị với phần tử [0,0] = 0
            (intercept không bị phạt).

    @input  X   : array-like, shape (n, p)
                  Ma trận đặc trưng (chưa có cột intercept).
            y   : array-like, shape (n,) hoặc (n, 1)
                  Vector biến mục tiêu.
            lam : float
                  Hệ số regularization lambda (>= 0).
                  Khi lam = 0 tương đương với OLS thông thường.

    @output beta_hat_ridge : np.ndarray, shape (p+1, 1)
                             Vector hệ số Ridge ước lượng (gồm intercept ở vị trí 0).

    @raises ValueError  Nếu ma trận (X^T X + lambda * I_mod) suy biến.
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
    @brief  Vẽ Ridge Trace — đồ thị biểu diễn sự thay đổi của các hệ số hồi quy
            (trừ intercept) theo giá trị lambda trên thang logarithm.
            Giúp quan sát quá trình co rút (shrinkage) hệ số khi lambda tăng.

    @input  X       : array-like, shape (n, p)
                      Ma trận đặc trưng (chưa có cột intercept).
            y       : array-like, shape (n,) hoặc (n, 1)
                      Vector biến mục tiêu.
            lambdas : array-like, shape (m,)
                      Danh sách các giá trị lambda cần khảo sát (nên trải đều
                      trên thang log, ví dụ np.logspace(-3, 5, 100)).

    @output None. Hiển thị figure matplotlib với đồ thị Ridge Trace.
                  Trục x: lambda (thang log), trục y: giá trị hệ số.
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
        plt.plot(lambdas, coefs[:, j], label=f'Feature {j + 1}')

    plt.xscale('log')  # Hiển thị thang log để dễ nhìn sự hội tụ
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Coefficients (Hệ số)')
    plt.title('Ridge Trace (Đồ thị thay đổi hệ số theo Lambda)')
    plt.legend()
    plt.grid(True)
    plt.show()
