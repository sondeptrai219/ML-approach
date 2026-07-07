import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_oos_r2(y_true, y_pred, y_train_mean):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)
    return 1 - (numerator / denominator)

def main():
    file_path = "data/fixed_dataset.xlsx"
    df = pd.read_excel(file_path)
    
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    unique_dates = sorted(df_clean['Date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(0.75 * n_dates)
    
    df_train = df_clean[df_clean['Date'].isin(unique_dates[:train_end_idx])]
    df_test = df_clean[df_clean['Date'].isin(unique_dates[int(0.875 * n_dates):])]
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test_raw = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    y_train_mean = y_train.mean()
    
    alphas = np.logspace(-3, 5, 100)
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2_standard = r2_score(y_test, y_pred)
    test_r2_oos = calculate_oos_r2(y_test, y_pred, y_train_mean)
    test_corr, _ = pearsonr(y_test, y_pred)
    
    print("\n================== RIDGE TEST PERFORMANCE ==================")
    print(f"MSE: {test_mse:.6f} | MAE: {test_mae:.6f}")
    print(f"R2: {test_r2_standard:.6%} | OOS R2: {test_r2_oos:.6%}")
    print(f"Correlation: {test_corr:.4f}")
    print("============================================================")

if __name__ == '__main__':
    main()
