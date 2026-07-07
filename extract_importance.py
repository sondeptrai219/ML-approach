import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

# Load and clean data
df = pd.read_excel("data/fixed_dataset.xlsx")
id_cols = ['ticker', 'Date', 'mapped_quarter']
target_col = 'Target'
feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]

df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)

# Split (Identical Train Split)
unique_dates = sorted(df_clean['Date'].unique())
train_end_idx = int(0.75 * len(unique_dates))
train_dates = unique_dates[:train_end_idx]
df_train = df_clean[df_clean['Date'].isin(train_dates)]

X_train_raw = df_train[feature_cols].values
y_train = df_train[target_col].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)

# 1. Re-fit the best models to get importances
print("Fitting Ridge model...")
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
ridge.fit(X_train, y_train)

print("Fitting Random Forest (max_depth=3) model...")
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 2. Extract Ridge Coefficients
ridge_coefs = ridge.coef_
ridge_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': ridge_coefs,
    'Abs_Coefficient': np.abs(ridge_coefs)
}).sort_values(by='Abs_Coefficient', ascending=False)

# 3. Extract RF Importances
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({
    'Feature': feature_cols,
    'Gini_Importance': rf_importances
}).sort_values(by='Gini_Importance', ascending=False)

# Print Top 15 Features for both
print("\n--- Top 15 Ridge Coefficients (Linear Signal Strength) ---")
print(ridge_df.head(15).to_string(index=False))

print("\n--- Top 15 Random Forest Features (Non-linear Split Value) ---")
print(rf_df.head(15).to_string(index=False))

# Save results
ridge_df.to_csv("results/ridge_coefficients.csv", index=False)
rf_df.to_csv("results/rf_importances.csv", index=False)
print("\nFeature importance results saved to CSV files.")
