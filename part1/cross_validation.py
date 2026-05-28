import numpy as np
from part1.ols_implementation import _add_intercept


def kfold_cv(X, y, k, model_func, random_state=42, **kwargs):
    """
    @brief  Thực hiện K-Fold Cross Validation từ đầu (không dùng sklearn).
            Dữ liệu được xáo trộn ngẫu nhiên rồi chia đều thành k phần (fold).
            Mỗi vòng lặp: dùng k-1 fold để huấn luyện và 1 fold còn lại để kiểm tra.
            Kết quả cuối cùng là trung bình MSE trên k lần kiểm tra.
            Công thức: CV(k) = (1/k) * sum_{i=1}^{k} MSE_i.

    @input  X            : array-like, shape (n, p)
                           Ma trận đặc trưng (chưa có cột intercept).

            y            : array-like, shape (n,) hoặc (n, 1)
                           Vector biến mục tiêu.

            k            : int
                           Số fold (thường dùng k = 5 hoặc k = 10).

            model_func   : callable
                           Hàm huấn luyện mô hình với signature:
                           f(X_train, y_train, **kwargs).

            random_state : int, mặc định = 42
                           Seed để đảm bảo reproducibility.

            **kwargs     : dict
                           Tham số bổ sung truyền vào model_func
                           (ví dụ: lam=1.0 cho Ridge).

    @output mean_mse : float
                       Giá trị trung bình MSE trên k fold.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    n = X.shape[0]

    # Random generator để reproducible
    rng = np.random.default_rng(random_state)

    # Shuffle indices
    indices = rng.permutation(n)

    # Chia indices thành k folds
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1

    current = 0
    folds = []

    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size

        folds.append(indices[start:stop])

        current = stop

    mse_list = []

    for i in range(k):

        # Fold i dùng làm test
        test_idx = folds[i]

        # Các fold còn lại làm train
        train_idx = np.concatenate([
            folds[j] for j in range(k) if j != i
        ])

        X_train = np.asarray(X[train_idx])
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        y_train = y[train_idx]

        X_test = np.asarray(X[test_idx])
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        y_test = y[test_idx]
        

        # Train model
        result = model_func(X_train, y_train, **kwargs)

        # OLS có thể trả tuple (beta_hat, sigma2)
        if isinstance(result, tuple):
            beta_hat = result[0]
        else:
            beta_hat = result

        beta_hat = np.asarray(beta_hat).reshape(-1, 1)

        # Predict
        X_test_int = _add_intercept(X_test)
        y_pred = X_test_int @ beta_hat

        # MSE
        mse = np.mean((y_test - y_pred) ** 2)

        mse_list.append(mse)

    return np.mean(mse_list)
