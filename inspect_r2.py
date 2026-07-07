import pandas as pd
import numpy as np
import torch
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.append("C:/Users/Asus/.gemini/antigravity/scratch/stock_fnn_predictor")
from model import StockPredictorFNN

def main():
    # 1. Load and Clean Dataset
    print("Loading dataset...")
    df = pd.read_excel("data/fixed_dataset.xlsx")
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    # Split
    unique_dates = sorted(df_clean['Date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(0.75 * n_dates)
    val_end_idx = int(0.875 * n_dates)
    
    df_train = df_clean[df_clean['Date'].isin(unique_dates[:train_end_idx])]
    df_test = df_clean[df_clean['Date'].isin(unique_dates[val_end_idx:])]
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test_raw = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 2. Get predictions on Train and Test sets
    print("\nTraining and predicting models...")
    
    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_train, y_train)
    ridge_pred_train = ridge.predict(X_train)
    ridge_pred_test = ridge.predict(X_test)
    
    # RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    
    # FNN
    fnn = StockPredictorFNN(len(feature_cols))
    fnn.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    fnn.eval()
    with torch.no_grad():
        fnn_pred_train = fnn(torch.tensor(X_train, dtype=torch.float32)).numpy().flatten()
        fnn_pred_test = fnn(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        
    # 3. Calculate R2 scores
    # Standard R2 is relative to the mean of the evaluated dataset (Train mean for Train R2, Test mean for Test R2)
    print("\n=== Standard R2 (scikit-learn defaults) ===")
    print(f"Ridge Train R2: {r2_score(y_train, ridge_pred_train):.6%}")
    print(f"Ridge Test R2:  {r2_score(y_test, ridge_pred_test):.6%}")
    print(f"FNN Train R2:   {r2_score(y_train, fnn_pred_train):.6%}")
    print(f"FNN Test R2:    {r2_score(y_test, fnn_pred_test):.6%}")
    print(f"RF Train R2:    {r2_score(y_train, rf_pred_train):.6%}")
    print(f"RF Test R2:     {r2_score(y_test, rf_pred_test):.6%}")
    
    # 4. Out-of-sample R2 (using Train mean as baseline)
    def oos_r2(y_true, y_pred, y_train_mean):
        num = np.sum((y_true - y_pred) ** 2)
        den = np.sum((y_true - y_train_mean) ** 2)
        return 1 - (num / den)
        
    y_train_mean = y_train.mean()
    print("\n=== Out-of-Sample R2 (Relative to Train Mean) ===")
    print(f"Ridge Test OOS R2: {oos_r2(y_test, ridge_pred_test, y_train_mean):.6%}")
    print(f"FNN Test OOS R2:  {oos_r2(y_test, fnn_pred_test, y_train_mean):.6%}")
    print(f"RF Test OOS R2:   {oos_r2(y_test, rf_pred_test, y_train_mean):.6%}")
    
    # 5. Examine Stock Returns Target Stats
    print("\n=== Target Statistics (Vietnam Stock Returns) ===")
    print(f"Train set mean return: {y_train.mean():.6f}")
    print(f"Test set mean return:  {y_test.mean():.6f} (Difference: {y_test.mean() - y_train.mean():.6f})")
    print(f"Train max return:      {y_train.max():.6f} | Min return: {y_train.min():.6f}")
    print(f"Test max return:       {y_test.max():.6f} | Min return: {y_test.min():.6f}")
    print(f"Train std deviation:   {y_train.std():.6f}")
    
if __name__ == '__main__':
    main()
