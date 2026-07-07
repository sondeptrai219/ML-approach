import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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

def train_fnn(model, train_loader, val_loader, device, epochs=100, patience=8):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                
        val_loss = running_val_loss / len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Restore best weights
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model, best_epoch

def train_fnn_combined(model, train_val_loader, device, epochs):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_val_loader:
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
    
    # Tuning configurations
    layer_configs = [
        [64, 32, 16],   # Standard
        [128, 64, 32],  # Wide
        [32, 16, 8]     # Narrow
    ]
    activations = ['ReLU', 'LeakyReLU', 'Swish']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    all_history = []
    final_test_results = []
    
    for split_name, (train_m, val_m, test_m) in splits_config.items():
        print(f"\n==========================================")
        print(f"         RUNNING FNN {split_name.upper()}  ")
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
        
        # Scale for Grid Search (Fit on Train, scale Val)
        scaler_gs = StandardScaler()
        X_train = scaler_gs.fit_transform(X_train_raw)
        X_val = scaler_gs.transform(X_val_raw)
        
        y_train_mean = y_train.mean()
        
        # Loaders for Grid Search
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        input_dim = len(feature_cols)
        
        split_history = []
        best_val_mse = float('inf')
        best_cfg_str = ""
        best_opt_epoch = 0
        best_layers = None
        best_act = None
        best_val_r2 = 0.0
        best_val_r2_oos = 0.0
        best_val_corr = 0.0
        
        print(f"Running grid search over 9 FNN configurations...")
        
        for layers in layer_configs:
            for act_name in activations:
                # Initialize model
                model = DynamicFNN(input_dim, layers, act_name).to(device)
                
                # Train model
                model, opt_epoch = train_fnn(model, train_loader, val_loader, device, epochs=100, patience=8)
                
                # Predict on validation set
                y_val_true, y_val_pred = get_predictions(model, val_loader, device)
                
                # Metrics
                val_mse = mean_squared_error(y_val_true, y_val_pred)
                val_mae = mean_absolute_error(y_val_true, y_val_pred)
                val_r2 = r2_score(y_val_true, y_val_pred)
                val_r2_oos = calculate_oos_r2(y_val_true, y_val_pred, y_train_mean)
                
                # Pearson correlation
                try:
                    val_corr, _ = pearsonr(y_val_true, y_val_pred)
                except Exception:
                    val_corr = np.nan
                    
                cfg_str = f"Layers: {layers}, Act: {act_name}"
                split_history.append({
                    'Layers': str(layers),
                    'Activation': act_name,
                    'Val_MSE': val_mse,
                    'Val_MAE': val_mae,
                    'Val_R2': val_r2,
                    'Val_R2_OOS': val_r2_oos,
                    'Val_Corr': val_corr,
                    'Opt_Epoch': opt_epoch
                })
                
                # Select best model based on Validation MSE
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_cfg_str = cfg_str
                    best_layers = layers
                    best_act = act_name
                    best_val_r2 = val_r2
                    best_val_r2_oos = val_r2_oos
                    best_val_corr = val_corr
                    best_opt_epoch = opt_epoch
                    
        # Save split history
        history_df = pd.DataFrame(split_history)
        history_df['Split'] = split_name
        all_history.append(history_df)
        
        print(f"\nOptimal FNN configuration for {split_name}: {best_cfg_str}")
        print(f"  Val MSE:       {best_val_mse:.6f}")
        print(f"  Val R2:        {best_val_r2:.6%}")
        print(f"  Val R2_OOS:    {best_val_r2_oos:.6%}")
        print(f"  Val Corr:      {best_val_corr:.4f}")
        print(f"  Optimal Epoch: {best_opt_epoch}")
        
        # --- REFITTING PHASE (Combine Train + Val) ---
        X_train_val_raw = np.vstack([X_train_raw, X_val_raw])
        y_train_val = np.concatenate([y_train, y_val])
        
        # Fit new scaler on combined data
        scaler_combined = StandardScaler()
        X_train_val = scaler_combined.fit_transform(X_train_val_raw)
        X_test = scaler_combined.transform(X_test_raw)
        
        # Loaders for combined data
        train_val_dataset = StockDataset(X_train_val, y_train_val)
        test_dataset_refit = StockDataset(X_test, y_test)
        
        train_val_loader = DataLoader(train_val_dataset, batch_size=256, shuffle=True)
        test_loader_refit = DataLoader(test_dataset_refit, batch_size=512, shuffle=False)
        
        # Initialize optimal model architecture
        refit_model = DynamicFNN(input_dim, best_layers, best_act).to(device)
        
        print(f"Refitting optimal FNN on combined Train+Val set (size: {len(X_train_val)}) for {best_opt_epoch} epochs...")
        refit_model = train_fnn_combined(refit_model, train_val_loader, device, epochs=best_opt_epoch)
        
        # Save best model weights (refitted)
        torch.save(refit_model.state_dict(), f'models/best_fnn_{split_name}.pth')
        
        # Evaluate Best FNN on Test set
        y_test_true, y_test_pred = get_predictions(refit_model, test_loader_refit, device)
        
        test_mse = mean_squared_error(y_test_true, y_test_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)
        test_r2_oos = calculate_oos_r2(y_test_true, y_test_pred, y_train_mean)
        try:
            test_corr, _ = pearsonr(y_test_true, y_test_pred)
        except Exception:
            test_corr = np.nan
            
        print(f"\nTest Set Performance (Refitted Model):")
        print(f"  MSE:              {test_mse:.6f}")
        print(f"  MAE:              {test_mae:.6f}")
        print(f"  Standard R2:      {test_r2:.6%}")
        print(f"  OOS R2:           {test_r2_oos:.6%}")
        print(f"  Pearson Corr:     {test_corr:.4f}")
        
        final_test_results.append({
            'Split': split_name,
            'Train_Months': train_m,
            'Layers': str(best_layers),
            'Activation': best_act,
            'Val_MSE': best_val_mse,
            'Val_R2': best_val_r2,
            'Val_R2_OOS': best_val_r2_oos,
            'Val_Corr': best_val_corr,
            'Test_MSE': test_mse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Test_R2_OOS': test_r2_oos,
            'Test_Corr': test_corr
        })
        
    # Export history
    combined_history = pd.concat(all_history, ignore_index=True)
    try:
        combined_history.to_csv("results/rolling_fnn_history.csv", index=False)
        print(f"\nAll FNN grid search history saved to 'results/rolling_fnn_history.csv'.")
    except PermissionError:
        print("Warning: Permission denied for 'results/rolling_fnn_history.csv'. Saving to 'results/rolling_fnn_history_fallback.csv' instead.")
        combined_history.to_csv("results/rolling_fnn_history_fallback.csv", index=False)
    
    # Export test results
    results_df = pd.DataFrame(final_test_results)
    try:
        results_df.to_csv("results/rolling_fnn_test_results.csv", index=False)
        print(f"Final FNN test results for all splits saved to 'results/rolling_fnn_test_results.csv'.")
    except PermissionError:
        print("Warning: Permission denied for 'results/rolling_fnn_test_results.csv'. Saving to 'results/rolling_fnn_test_results_fallback.csv' instead.")
        results_df.to_csv("results/rolling_fnn_test_results_fallback.csv", index=False)
    
    # Print summary table
    print("\n================== ROLLING FNN FINAL TEST PERFORMANCE ==================")
    print(results_df[[
        'Split', 'Layers', 'Activation', 'Test_MSE', 'Test_R2', 'Test_R2_OOS', 'Test_Corr'
    ]].to_string(index=False, formatters={
        'Test_MSE': lambda x: f"{x:.6f}",
        'Test_R2': lambda x: f"{x:.2%}",
        'Test_R2_OOS': lambda x: f"{x:.2%}",
        'Test_Corr': lambda x: f"{x:.4f}"
    }))
    print("=======================================================================")

if __name__ == '__main__':
    main()
