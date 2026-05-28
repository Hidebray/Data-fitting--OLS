import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, target_col='CO(GT)', imputation_method='knn'):
        """
        @brief Khởi tạo chuỗi tiền xử lý dữ liệu.
        @input target_col (str): Tên cột mục tiêu.
        @input imputation_method (str): Phương pháp điền khuyết.
        """
        self.target_col = target_col
        self.imputation_method = imputation_method
        self.imputer = KNNImputer(n_neighbors=5) if imputation_method == 'knn' else None
        self.scaler = StandardScaler()
        
        self.feature_cols = None
        self.lower_bounds = None
        self.upper_bounds = None
        
    def fit(self, X_train):
        """
        @brief Lưu cấu trúc, fit Imputer và Scaler trên tập huấn luyện (chống data leakage).
        @input X_train (pd.DataFrame hoặc np.ndarray): Dữ liệu huấn luyện.
        @output self: Đối tượng DataPipeline đã được fit.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_cols = X_train.columns.tolist()
            X_vals = X_train.values
        else:
            self.feature_cols = None
            X_vals = X_train
            
        if self.imputer:
            self.imputer.fit(X_vals)
            X_tmp = self.imputer.transform(X_vals)
        else:
            X_tmp = np.copy(X_vals)
            
        self.lower_bounds = np.percentile(X_tmp, 1, axis=0)
        self.upper_bounds = np.percentile(X_tmp, 99, axis=0)
        X_tmp = np.clip(X_tmp, self.lower_bounds, self.upper_bounds)
        
        self.scaler.fit(X_tmp)
        return self
        
    def transform(self, X):
        """
        @brief Thực hiện chuỗi biến đổi trên tập dữ liệu.
        @input X (pd.DataFrame hoặc np.ndarray): Dữ liệu cần biến đổi.
        @output np.ndarray: Dữ liệu đã qua xử lý, được thêm cột Intercept.
        """
        if isinstance(X, pd.DataFrame):
            X_vals = X[self.feature_cols].values if self.feature_cols else X.values
        else:
            X_vals = np.copy(X)
            
        if self.imputer:
            X_transformed = self.imputer.transform(X_vals)
        else:
            X_transformed = np.copy(X_vals)
            
        X_transformed = np.clip(X_transformed, self.lower_bounds, self.upper_bounds)
        X_transformed = self.scaler.transform(X_transformed)
        
        ones = np.ones((X_transformed.shape[0], 1))
        X_transformed = np.hstack((ones, X_transformed))
        
        return X_transformed
        
    def fit_transform(self, X_train):
        """
        @brief Fit và Transform đồng thời trên tập huấn luyện.
        @input X_train (pd.DataFrame hoặc np.ndarray): Dữ liệu huấn luyện.
        @output np.ndarray: Dữ liệu huấn luyện đã biến đổi.
        """
        self.fit(X_train)
        return self.transform(X_train)

def load_and_preprocess_raw_data(filepath, target_col='CO(GT)'):
    """
    @brief Đọc dữ liệu từ CSV, thay thế missing value, và loại bỏ cột/hàng lỗi.
    @input filepath (str): Đường dẫn đến file dữ liệu.
    @input target_col (str): Tên cột mục tiêu.
    @output tuple: (X, y) dưới dạng pandas DataFrame/Series.
    @raises FileNotFoundError: Nếu không tìm thấy file.
    @raises ValueError: Nếu không tìm thấy cột target.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file tại {filepath}")
        
    df = pd.read_csv(filepath, sep=';', decimal=',')
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    df.replace(-200, np.nan, inplace=True)
    
    if 'NMHC(GT)' in df.columns:
        df.drop(columns=['NMHC(GT)'], inplace=True)
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)
    if 'Time' in df.columns:
        df.drop(columns=['Time'], inplace=True)
        
    if target_col not in df.columns:
        raise ValueError(f"Lỗi: Không tìm thấy cột Target '{target_col}' trong dataset.")
        
    df.dropna(subset=[target_col], inplace=True)
    
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col]).astype(float)
    
    return X, y