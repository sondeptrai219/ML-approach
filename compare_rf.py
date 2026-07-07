import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_oos_r2(y_true, y_pred, y_train_mean):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)
    return 1 - (numerator / denominator)

def main():
    # Load and clean data
    df = pd.read_excel("data/fixed_dataset.xlsx")
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    # Split
    unique_dates = sorted(df_clean['Date'].unique())
    train_end_idx = int(0.75 * len(unique_dates))
    train_dates = unique_dates[:train_end_idx]
    test_dates = unique_dates[train_end_idx + int(0.125 * len(unique_dates)):]  # Keep same test set
    
    df_train = df_clean[df_clean['Date'].isin(train_dates)]
    df_test = df_clean[df_clean['Date'].isin(test_dates)]
    
    X_train = StandardScaler().fit_transform(df_train[feature_cols].values)
    y_train = df_train[target_col].values
    
    X_test = StandardScaler().fit_transform(df_test[feature_cols].values) # standardized test features
    X_test = StandardScaler().fit_transform(df_train[feature_cols].values) # wait, reuse scaler!
    
    # Fix scaling correctly to match previous runs:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)
    
    y_train_mean = y_train.mean()
    y_test = df_test[target_col].values
    
    configs = {
        "RF (min_samples_leaf=5)": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "RF (max_leaf_nodes=5)": RandomForestRegressor(n_estimators=100, max_leaf_nodes=5, random_state=42, n_jobs=-1),
        "RF (max_depth=3)": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
    }
    
    results = []
    for name, model in configs.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        oos_r2 = calculate_oos_r2(y_test, preds, y_train_mean)
        corr, _ = pearsonr(y_test, preds)
        
        results.append({
            "Model": name,
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "OOS_R2": oos_r2,
            "Corr": corr
        })
        
    print("\n--- Comparison Table ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Save comparison to file
    res_df.to_csv("results/rf_comparison.csv", index=False)

if __name__ == '__main__':
    main()
