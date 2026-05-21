import numpy as np
from part1.ols_implementation import _add_intercept

def kfold_cv(X, y, k, model_func, **kwargs):
    """
    Tự cài đặt logic K-Fold Cross Validation.
    Lặp k lần: huấn luyện k-1 fold, dự đoán 1 fold. Trả về MSE trung bình.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    n = X.shape[0]
    
    # Trộn ngẫu nhiên index
    indices = np.random.permutation(n)
    
    # Chia mảng indices thành k phần (k-folds)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
        
    mse_list = []
    
    for i in range(k):
        # Lấy fold thứ i làm tập Test
        test_idx = folds[i]
        
        # Các fold còn lại làm tập Train
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Train mô hình trên tập Train (truyền thêm **kwargs cho tham số phụ như lam)
        result = model_func(X_train, y_train, **kwargs)
        
        # Xử lý trường hợp OLS trả về (beta_hat, sigma_squared) thay vì chỉ beta_hat
        if isinstance(result, tuple):
            beta_hat = result[0]
        else:
            beta_hat = result
            
        # Dự đoán trên tập Test
        X_test_int = _add_intercept(X_test)
        y_pred = X_test_int @ beta_hat
        
        # Tính Mean Squared Error cho fold hiện tại
        mse = np.mean((y_test - y_pred) ** 2)
        mse_list.append(mse)
        
    return np.mean(mse_list)

if __name__ == '__main__':
    from part1.ols_implementation import ols_fit
    from part1.ridge_lasso import ridge_fit
    
    np.random.seed(42)
    X_test = np.random.randn(100, 3)
    y_test = X_test @ np.array([[1.5], [-2.0], [0.5]]) + np.random.randn(100, 1) * 0.5
    
    mse_ols = kfold_cv(X_test, y_test, k=5, model_func=ols_fit)
    print("Trung bình MSE của OLS qua 5-Fold CV:", mse_ols)
    
    mse_ridge = kfold_cv(X_test, y_test, k=5, model_func=ridge_fit, lam=10.0)
    print("Trung bình MSE của Ridge (lambda=10.0) qua 5-Fold CV:", mse_ridge)
