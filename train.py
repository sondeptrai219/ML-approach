import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Import custom dataset and model
import sys
sys.path.append("C:/Users/Asus/.gemini/antigravity/scratch/stock_fnn_predictor")
from model import StockDataset, StockPredictorFNN

torch.manual_seed(42)
np.random.seed(42)

def calculate_oos_r2(y_true, y_pred, y_train_mean):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_train_mean) ** 2)
    return 1 - (numerator / denominator)

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
    
    unique_dates = sorted(df_clean['Date'].unique())
    n_dates = len(unique_dates)
    
    train_end_idx = int(0.75 * n_dates)
    val_end_idx = int(0.875 * n_dates)
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    df_train = df_clean[df_clean['Date'].isin(train_dates)]
    df_val = df_clean[df_clean['Date'].isin(val_dates)]
    df_test = df_clean[df_clean['Date'].isin(test_dates)]
    
    X_train_raw = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_val_raw = df_val[feature_cols].values
    y_val = df_val[target_col].values
    X_test_raw = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    y_train_mean = y_train.mean()
    
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    input_dim = len(feature_cols)
    model = StockPredictorFNN(input_dim)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    epochs = 200
    patience = 15
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    def get_all_preds_and_trues(loader):
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_trues.extend(batch_y.numpy().flatten())
        return np.array(all_trues), np.array(all_preds)
        
    y_test_true, y_test_pred = get_all_preds_and_trues(test_loader)
    
    test_mse = mean_squared_error(y_test_true, y_test_pred)
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_r2_standard = r2_score(y_test_true, y_test_pred)
    test_r2_oos = calculate_oos_r2(y_test_true, y_test_pred, y_train_mean)
    test_corr, _ = pearsonr(y_test_true, y_test_pred)
    
    print("\n================== FNN TEST PERFORMANCE ==================")
    print(f"MSE: {test_mse:.6f} | MAE: {test_mae:.6f}")
    print(f"R2: {test_r2_standard:.6%} | OOS R2: {test_r2_oos:.6%}")
    print(f"Correlation: {test_corr:.4f}")
    print("==========================================================")

if __name__ == '__main__':
    main()
