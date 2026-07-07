import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DynamicFNN(nn.Module):
    def __init__(self, input_dim, layers, activation_name):
        super(DynamicFNN, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Map activation name to PyTorch activation module
        if activation_name == 'ReLU':
            act_fn = nn.ReLU
        elif activation_name == 'LeakyReLU':
            act_fn = nn.LeakyReLU
        elif activation_name == 'Swish':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
            
        self.layer1 = nn.Linear(input_dim, layers[0])
        self.bn1 = nn.BatchNorm1d(layers[0])
        self.act1 = act_fn()
        self.drop1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(layers[0], layers[1])
        self.bn2 = nn.BatchNorm1d(layers[1])
        self.act2 = act_fn()
        self.drop2 = nn.Dropout(0.05)
        
        self.layer3 = nn.Linear(layers[1], layers[2])
        self.bn3 = nn.BatchNorm1d(layers[2])
        self.act3 = act_fn()
        self.drop3 = nn.Dropout(0.0)
        
        self.output_layer = nn.Linear(layers[2], 1)

    def forward(self, x):
        x = self.input_bn(x)
        
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        
        out = self.output_layer(x)
        return out

def calculate_oos_r2(y_true, y_pred, y_train_mean):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)
    if denominator == 0:
        return 0.0
    return 1.0 - (numerator / denominator)

def train_fnn_fixed(model, train_loader, device, epochs):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_trues.extend(batch_y.numpy().flatten())
    return np.array(all_trues), np.array(all_preds)

def main():
    file_path = "data/fixed_dataset.xlsx"
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    # Chronological Split
    df_train = df_clean[(df_clean['Date'] >= '2023-01-01') & (df_clean['Date'] < '2025-05-01')]
    df_test = df_clean[(df_clean['Date'] >= '2025-05-01') & (df_clean['Date'] <= '2026-05-01')]
    
    print(f"Train months: {df_train['Date'].nunique()} (Rows: {len(df_train)})")
    print(f"Test months: {df_test['Date'].nunique()} (Rows: {len(df_test)})")
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test_raw = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    y_train_mean = y_train.mean()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for deep learning: {device}")
    
    results = []
    
    # ==========================================
    # 1. Ridge Baseline
    # ==========================================
    print("\nRunning Ridge Regression...")
    alphas = np.logspace(-3, 5, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train, y_train)
    ridge_preds = ridge_cv.predict(X_test)
    
    ridge_mse = mean_squared_error(y_test, ridge_preds)
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_r2 = r2_score(y_test, ridge_preds)
    ridge_oos = calculate_oos_r2(y_test, ridge_preds, y_train_mean)
    try:
        ridge_corr, _ = pearsonr(y_test, ridge_preds)
    except Exception:
        ridge_corr = np.nan
        
    results.append({
        'Model': 'Ridge Regression',
        'Config/Parameters': f'alpha={ridge_cv.alpha_:.4f}',
        'Test_MSE': ridge_mse,
        'Test_MAE': ridge_mae,
        'Test_R2': ridge_r2,
        'Test_R2_OOS': ridge_oos,
        'Test_Corr': ridge_corr
    })
    
    # ==========================================
    # 2. Random Forest
    # ==========================================
    print("Running Random Forest...")
    rf_depths = [2, 3, 5, None]
    for depth in rf_depths:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        
        rf_mse = mean_squared_error(y_test, rf_preds)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_r2 = r2_score(y_test, rf_preds)
        rf_oos = calculate_oos_r2(y_test, rf_preds, y_train_mean)
        try:
            rf_corr, _ = pearsonr(y_test, rf_preds)
        except Exception:
            rf_corr = np.nan
            
        depth_str = str(depth) if depth is not None else 'None'
        results.append({
            'Model': 'Random Forest',
            'Config/Parameters': f'n_est=100, depth={depth_str}, min_leaf=2',
            'Test_MSE': rf_mse,
            'Test_MAE': rf_mae,
            'Test_R2': rf_r2,
            'Test_R2_OOS': rf_oos,
            'Test_Corr': rf_corr
        })

    # ==========================================
    # 3. Feedforward Neural Network
    # ==========================================
    print("Running Feedforward Neural Networks...")
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    fnn_layers = [[32, 16, 8], [64, 32, 16]]
    fnn_activations = ['ReLU', 'LeakyReLU', 'Swish']
    fnn_epochs = [15, 30]
    
    for layers in fnn_layers:
        for act in fnn_activations:
            for ep in fnn_epochs:
                # Reset seeds before initializing each network to ensure comparative determinism
                np.random.seed(42)
                torch.manual_seed(42)
                
                model = DynamicFNN(input_dim, layers, act).to(device)
                model = train_fnn_fixed(model, train_loader, device, epochs=ep)
                
                y_true_fnn, y_pred_fnn = get_predictions(model, test_loader, device)
                
                fnn_mse = mean_squared_error(y_true_fnn, y_pred_fnn)
                fnn_mae = mean_absolute_error(y_true_fnn, y_pred_fnn)
                fnn_r2 = r2_score(y_true_fnn, y_pred_fnn)
                fnn_oos = calculate_oos_r2(y_true_fnn, y_pred_fnn, y_train_mean)
                try:
                    fnn_corr, _ = pearsonr(y_true_fnn, y_pred_fnn)
                except Exception:
                    fnn_corr = np.nan
                    
                results.append({
                    'Model': 'FNN',
                    'Config/Parameters': f'Layers={layers}, Act={act}, Epochs={ep}',
                    'Test_MSE': fnn_mse,
                    'Test_MAE': fnn_mae,
                    'Test_R2': fnn_r2,
                    'Test_R2_OOS': fnn_oos,
                    'Test_Corr': fnn_corr
                })
                
    # ==========================================
    # Print Results
    # ==========================================
    results_df = pd.DataFrame(results)
    
    # Save results to a CSV
    try:
        results_df.to_csv("results/custom_train_test_results.csv", index=False)
        print("\nResults exported to 'results/custom_train_test_results.csv'.")
    except PermissionError:
        print("\nWarning: Permission denied for 'results/custom_train_test_results.csv'. Saving to 'results/custom_train_test_results_fallback.csv' instead.")
        results_df.to_csv("results/custom_train_test_results_fallback.csv", index=False)
        
    print("\n" + "="*45 + " CUSTOM RUN PERFORMANCE (2023-2025 Train -> 2025-2026 Test) " + "="*45)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Format and print Ridge & RF
    print("\n--- Linear & Tree Models ---")
    print(results_df[results_df['Model'] != 'FNN'].to_string(index=False, formatters={
        'Test_MSE': lambda x: f"{x:.6f}",
        'Test_MAE': lambda x: f"{x:.6f}",
        'Test_R2': lambda x: f"{x:.2%}",
        'Test_R2_OOS': lambda x: f"{x:.2%}",
        'Test_Corr': lambda x: f"{x:.4f}"
    }))
    
    # Format and print FNN
    print("\n--- Feedforward Neural Networks ---")
    print(results_df[results_df['Model'] == 'FNN'].to_string(index=False, formatters={
        'Test_MSE': lambda x: f"{x:.6f}",
        'Test_MAE': lambda x: f"{x:.6f}",
        'Test_R2': lambda x: f"{x:.2%}",
        'Test_R2_OOS': lambda x: f"{x:.2%}",
        'Test_Corr': lambda x: f"{x:.4f}"
    }))
    print("="*120)

if __name__ == '__main__':
    main()
