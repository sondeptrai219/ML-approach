import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import torch

import sys
import os
sys.path.append("C:/Users/Asus/.gemini/antigravity/scratch/stock_fnn_predictor")

# Import FNN model structure
from model import StockPredictorFNN

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

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
    train_end_idx = int(0.75 * len(unique_dates))
    val_end_idx = int(0.875 * len(unique_dates))
    
    df_train = df_clean[df_clean['Date'].isin(unique_dates[:train_end_idx])]
    df_test = df_clean[df_clean['Date'].isin(unique_dates[val_end_idx:])]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    # 2. Fit Ridge Regression
    print("Fitting Ridge model...")
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_train, y_train)
    ridge_imp = np.abs(ridge.coef_)
    
    # 3. Fit Random Forests
    print("Fitting RF (min_samples_leaf=5)...")
    rf_min = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_min.fit(X_train, y_train)
    rf_min_imp = rf_min.feature_importances_
    
    print("Fitting RF (max_leaf_nodes=5)...")
    rf_leaf = RandomForestRegressor(n_estimators=100, max_leaf_nodes=5, random_state=42, n_jobs=-1)
    rf_leaf.fit(X_train, y_train)
    rf_leaf_imp = rf_leaf.feature_importances_
    
    print("Fitting RF (max_depth=3)...")
    rf_depth = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    rf_depth.fit(X_train, y_train)
    rf_depth_imp = rf_depth.feature_importances_
    
    # 4. Load FNN and Calculate Permutation Importance
    print("Loading FNN weights & computing permutation importance...")
    fnn = StockPredictorFNN(len(feature_cols))
    fnn.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    fnn.eval()
    
    # Baseline FNN prediction
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        baseline_preds = fnn(X_test_tensor).numpy().flatten()
    baseline_mse = np.mean((y_test - baseline_preds) ** 2)
    
    fnn_imp = []
    for i in range(len(feature_cols)):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        X_permuted_tensor = torch.tensor(X_test_permuted, dtype=torch.float32)
        with torch.no_grad():
            permuted_preds = fnn(X_permuted_tensor).numpy().flatten()
        permuted_mse = np.mean((y_test - permuted_preds) ** 2)
        fnn_imp.append(max(0, permuted_mse - baseline_mse)) # keep importance >= 0
    fnn_imp = np.array(fnn_imp)
    
    # 5. Build Importance DataFrame
    imp_df = pd.DataFrame(index=feature_cols)
    imp_df['Ridge'] = ridge_imp
    imp_df['RF (min_leaf=5)'] = rf_min_imp
    imp_df['RF (max_leaf=5)'] = rf_leaf_imp
    imp_df['RF (max_depth=3)'] = rf_depth_imp
    imp_df['FNN'] = fnn_imp
    
    # 6. Column-wise Normalization (Min-Max Scaling to [0, 1] range)
    normalized_df = pd.DataFrame(index=feature_cols)
    for col in imp_df.columns:
        col_min = imp_df[col].min()
        col_max = imp_df[col].max()
        if col_max - col_min > 0:
            normalized_df[col] = (imp_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0.0
            
    # Calculate row-wise mean and sort features descending
    normalized_df['Mean_Importance'] = normalized_df.mean(axis=1)
    normalized_df = normalized_df.sort_values(by='Mean_Importance', ascending=False)
    
    # Save values to CSV
    normalized_df.to_csv("results/normalized_characteristic_importances.csv")
    
    # Drop mean column for plotting
    plot_df = normalized_df.drop(columns=['Mean_Importance'])
    
    # 7. Generate Heatmap Plot
    plt.figure(figsize=(9, 14))
    
    # We use a beautiful blue color palette ('Blues') with white grid lines separating cells
    ax = sns.heatmap(
        plot_df, 
        cmap='Blues', 
        cbar=True, 
        cbar_kws={'label': 'Normalized Importance (Relative within Model)'}, 
        linewidths=0.5, 
        linecolor='white',
        xticklabels=True,
        yticklabels=True
    )
    
    # Style customization
    plt.title('Figure 5: Characteristic Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=12, labelpad=12)
    plt.ylabel('Stock Characteristics / Financial Variables', fontsize=12, labelpad=12)
    
    # Align labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/characteristic_importance_heatmap.png', dpi=300)
    plt.close()
    
    print("\nHeatmap saved successfully to 'plots/characteristic_importance_heatmap.png'.")

if __name__ == '__main__':
    main()
