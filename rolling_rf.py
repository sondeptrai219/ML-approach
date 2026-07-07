import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Set random seed
np.random.seed(42)

def calculate_oos_r2(y_true, y_pred, y_train_mean):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)
    if denominator == 0:
        return 0.0
    return 1.0 - (numerator / denominator)

def run_grid_search(X_train, y_train, X_val, y_val, y_train_mean):
    # Hyperparameter Grid
    n_estimators_list = [50, 100, 200]
    max_depth_list = [2, 3, 5, None]
    min_samples_leaf_list = [2, 5, 10]
    
    trials = []
    
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            for min_leaf in min_samples_leaf_list:
                # Initialize model
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Fit model on training set
                rf.fit(X_train, y_train)
                
                # Predict on validation set
                preds = rf.predict(X_val)
                
                # Calculate metrics on validation set
                val_mse = mean_squared_error(y_val, preds)
                val_mae = mean_absolute_error(y_val, preds)
                val_r2 = r2_score(y_val, preds)
                val_r2_oos = calculate_oos_r2(y_val, preds, y_train_mean)
                val_corr, _ = pearsonr(y_val, preds)
                
                trials.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'min_samples_leaf': min_leaf,
                    'Val_MSE': val_mse,
                    'Val_MAE': val_mae,
                    'Val_R2': val_r2,
                    'Val_R2_OOS': val_r2_oos,
                    'Val_Corr': val_corr
                })
                
    # Sort by Val_MSE in ascending order
    trials_df = pd.DataFrame(trials)
    trials_df_sorted = trials_df.sort_values(by='Val_MSE', ascending=True).reset_index(drop=True)
    
    best_trial = trials_df_sorted.iloc[0]
    return best_trial, trials_df_sorted

def main():
    # 1. Load and Clean data
    file_path = "data/fixed_dataset.xlsx"
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    unique_dates = sorted(df_clean['Date'].unique())
    n_dates = len(unique_dates)
    print(f"Total months in dataset: {n_dates} (Cleaned rows: {len(df_clean)})")
    
    # 4 Splits configuration
    splits_config = {
        'Split_1': (36, 12, 24),
        'Split_2': (48, 12, 24),
        'Split_3': (60, 12, 24),
        'Split_4': (72, 12, 12)
    }
    
    all_history = []
    final_test_results = []
    
    for split_name, (train_m, val_m, test_m) in splits_config.items():
        print(f"\n==========================================")
        print(f"         RUNNING RF {split_name.upper()}  ")
        print(f"==========================================")
        
        train_start = 0
        train_end = train_m
        val_end = train_end + val_m
        test_end = val_end + test_m
        
        train_dates = unique_dates[train_start:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:test_end]
        
        df_train = df_clean[df_clean['Date'].isin(train_dates)]
        df_val = df_clean[df_clean['Date'].isin(val_dates)]
        df_test = df_clean[df_clean['Date'].isin(test_dates)]
        
        X_train_raw = df_train[feature_cols].values
        y_train = df_train[target_col].values
        X_val_raw = df_val[feature_cols].values
        y_val = df_val[target_col].values
        X_test_raw = df_test[feature_cols].values
        y_test = df_test[target_col].values
        
        # Scaling for Grid Search (Fit on Train, scale Val)
        scaler_gs = StandardScaler()
        X_train = scaler_gs.fit_transform(X_train_raw)
        X_val = scaler_gs.transform(X_val_raw)
        
        y_train_mean = y_train.mean()
        
        # Run grid search on Validation Set
        print(f"Running grid search over 36 parameter combinations...")
        best_trial, history_df = run_grid_search(X_train, y_train, X_val, y_val, y_train_mean)
        
        history_df['Split'] = split_name
        all_history.append(history_df)
        
        print(f"\nOptimal Parameters for {split_name} (Val MSE = {best_trial['Val_MSE']:.6f}):")
        print(f"  n_estimators:     {best_trial['n_estimators']}")
        print(f"  max_depth:        {best_trial['max_depth']}")
        print(f"  min_samples_leaf: {best_trial['min_samples_leaf']}")
        
        # --- REFITTING PHASE (Combine Train + Val) ---
        X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
        y_train_val = np.concatenate([y_train, y_val])
        
        # Fit new scaler on combined data
        scaler_combined = StandardScaler()
        X_train_val = scaler_combined.fit_transform(X_train_val_raw)
        X_test = scaler_combined.transform(X_test_raw)
        
        # Re-initialize best model
        best_rf = RandomForestRegressor(
            n_estimators=int(best_trial['n_estimators']),
            max_depth=int(best_trial['max_depth']) if pd.notnull(best_trial['max_depth']) else None,
            min_samples_leaf=int(best_trial['min_samples_leaf']),
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Refitting optimal RF on combined Train+Val set (size: {len(X_train_val)})...")
        best_rf.fit(X_train_val, y_train_val)
        
        # Evaluate on Test Set
        test_preds = best_rf.predict(X_test)
        
        test_mse = mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_r2_oos = calculate_oos_r2(y_test, test_preds, y_train_mean)
        try:
            test_corr, _ = pearsonr(y_test, test_preds)
        except Exception:
            test_corr = np.nan
            
        print(f"Test Set Performance (Refitted Model):")
        print(f"  MSE:              {test_mse:.6f}")
        print(f"  MAE:              {test_mae:.6f}")
        print(f"  Standard R2:      {test_r2:.6%}")
        print(f"  OOS R2:           {test_r2_oos:.6%}")
        print(f"  Pearson Corr:     {test_corr:.4f}")
        
        final_test_results.append({
            'Split': split_name,
            'Train_Months': train_m,
            'n_estimators': best_trial['n_estimators'],
            'max_depth': best_trial['max_depth'],
            'min_samples_leaf': best_trial['min_samples_leaf'],
            'Val_MSE': best_trial['Val_MSE'],
            'Val_R2': best_trial['Val_R2'],
            'Val_R2_OOS': best_trial['Val_R2_OOS'],
            'Val_Corr': best_trial['Val_Corr'],
            'Test_MSE': test_mse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Test_R2_OOS': test_r2_oos,
            'Test_Corr': test_corr
        })
        
    combined_history = pd.concat(all_history, ignore_index=True)
    try:
        combined_history.to_csv("results/rolling_rf_history.csv", index=False)
        print("RF grid search history saved to 'results/rolling_rf_history.csv'.")
    except PermissionError:
        print("Warning: Permission denied for 'results/rolling_rf_history.csv'. Saving to 'results/rolling_rf_history_fallback.csv' instead.")
        combined_history.to_csv("results/rolling_rf_history_fallback.csv", index=False)
    
    results_df = pd.DataFrame(final_test_results)
    try:
        results_df.to_csv("results/rolling_rf_test_results.csv", index=False)
        print("RF final test results saved to 'results/rolling_rf_test_results.csv'.")
    except PermissionError:
        print("Warning: Permission denied for 'results/rolling_rf_test_results.csv'. Saving to 'results/rolling_rf_test_results_fallback.csv' instead.")
        results_df.to_csv("results/rolling_rf_test_results_fallback.csv", index=False)
    
    print("\n================== ROLLING RF FINAL TEST PERFORMANCE (REFITTED) ==================")
    print(results_df[[
        'Split', 'n_estimators', 'max_depth', 'min_samples_leaf', 'Test_MSE', 'Test_R2', 'Test_R2_OOS', 'Test_Corr'
    ]].to_string(index=False, formatters={
        'Test_MSE': lambda x: f"{x:.6f}",
        'Test_R2': lambda x: f"{x:.2%}",
        'Test_R2_OOS': lambda x: f"{x:.2%}",
        'Test_Corr': lambda x: f"{x:.4f}"
    }))
    print("==================================================================================")

if __name__ == '__main__':
    main()
