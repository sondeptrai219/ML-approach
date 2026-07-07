import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed
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

def run_backtest_raw(df_test, pred_col, target_col, pct=0.10):
    """
    Simulate a Long-Short strategy without frictions.
    """
    dates = sorted(df_test['Date'].unique())
    portfolio_returns = []
    
    for date in dates:
        month_data = df_test[df_test['Date'] == date].copy()
        n_stocks = len(month_data)
        k = max(1, int(n_stocks * pct))
        
        # Sort by predicted return
        month_sorted = month_data.sort_values(by=pred_col, ascending=False)
        
        long_returns = month_sorted.head(k)[target_col].mean()
        short_returns = month_sorted.tail(k)[target_col].mean()
        
        ls_return = long_returns - short_returns
        portfolio_returns.append(ls_return)
        
    return np.array(portfolio_returns)

def calculate_metrics(returns):
    cum_returns = np.cumprod(1 + returns) - 1
    ann_return = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Max Drawdown
    peak = 1.0
    drawdowns = []
    running_wealth = 1.0
    for r in returns:
        running_wealth *= (1 + r)
        if running_wealth > peak:
            peak = running_wealth
        dd = (running_wealth - peak) / peak
        drawdowns.append(dd)
    max_dd = np.min(drawdowns)
    
    return {
        'Ann_Return': ann_return,
        'Ann_Vol': ann_vol,
        'Sharpe': sharpe,
        'Max_DD': max_dd,
        'Cum_Return': running_wealth - 1
    }

def main():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_excel("data/fixed_dataset.xlsx")
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    # Date splitting
    unique_dates = sorted(df_clean['Date'].unique())
    train_dates = unique_dates[:84]  # Split 4 Train + Val (Months 1–84)
    test_dates = unique_dates[84:]   # Split 4 Test (Months 85–96)
    
    df_train_val = df_clean[df_clean['Date'].isin(train_dates)].copy()
    df_test = df_clean[df_clean['Date'].isin(test_dates)].copy()
    
    print(f"Train+Val shape: {df_train_val.shape}")
    print(f"Test shape: {df_test.shape}")
    
    # 2. Scale Features
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(df_train_val[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)
    y_train_val = df_train_val[target_col].values
    
    # 3. Fit Models on Train+Val Set
    print("\nFitting models...")
    
    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_train_val, y_train_val)
    df_test['Pred_Ridge'] = ridge.predict(X_test)
    
    # Random Forest (Optimal Split 4: n_estimators=50, max_depth=2, min_samples_leaf=2)
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_val, y_train_val)
    df_test['Pred_RF'] = rf.predict(X_test)
    
    # FNN (Optimal Split 4 refitted: [64, 32, 16] LeakyReLU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(feature_cols)
    fnn = DynamicFNN(input_dim, [64, 32, 16], 'LeakyReLU').to(device)
    
    if os.path.exists('best_fnn_Split_4.pth'):
        print("Loading optimal FNN weights from 'best_fnn_Split_4.pth'...")
        fnn.load_state_dict(torch.load('best_fnn_Split_4.pth', map_location=device))
    else:
        print("Warning: 'best_fnn_Split_4.pth' not found. Training FNN for 15 epochs...")
        train_dataset = StockDataset(X_train_val, y_train_val)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.AdamW(fnn.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        fnn.train()
        for epoch in range(15):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(fnn(bx), by)
                loss.backward()
                optimizer.step()
                
    fnn.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        df_test['Pred_FNN'] = fnn(X_test_tensor).cpu().numpy().flatten()
        
    # Benchmark: Equal-weighted Market Return
    market_returns = []
    dates = sorted(df_test['Date'].unique())
    for date in dates:
        market_returns.append(df_test[df_test['Date'] == date][target_col].mean())
    market_returns = np.array(market_returns)
    
    # 4. Run Raw Backtests
    pct = 0.10
    ridge_ls = run_backtest_raw(df_test, 'Pred_Ridge', target_col, pct)
    rf_ls = run_backtest_raw(df_test, 'Pred_RF', target_col, pct)
    fnn_ls = run_backtest_raw(df_test, 'Pred_FNN', target_col, pct)
    
    # Compute metrics
    m_market = calculate_metrics(market_returns)
    m_ridge = calculate_metrics(ridge_ls)
    m_rf = calculate_metrics(rf_ls)
    m_fnn = calculate_metrics(fnn_ls)
    
    # Print results summary table
    backtest_data = [
        {'Strategy': 'Market Benchmark (Long)', **m_market},
        {'Strategy': 'Ridge Long-Short (Raw)', **m_ridge},
        {'Strategy': 'FNN Long-Short (Raw)', **m_fnn},
        {'Strategy': 'Random Forest L-S (Raw)', **m_rf}
    ]
    results_df = pd.DataFrame(backtest_data)
    
    print("\n" + "="*45 + " RAW BACKTEST PERFORMANCE (NO FRICTIONS) " + "="*45)
    print(results_df.to_string(index=False, formatters={
        'Ann_Return': lambda x: f"{x:.2%}",
        'Ann_Vol': lambda x: f"{x:.2%}",
        'Sharpe': lambda x: f"{x:.2f}",
        'Max_DD': lambda x: f"{x:.2%}",
        'Cum_Return': lambda x: f"{x:.2%}"
    }))
    print("="*125)
    
    # Save CSV
    results_df.to_csv("results/backtest_metrics_raw.csv", index=False)
    
    # 5. Plot Wealth Index Curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    dates_str = [d.strftime('%Y-%m') for d in dates]
    
    # Define colors
    c_market = '#7f8c8d'
    c_ridge = '#9b59b6'
    c_fnn = '#1abc9c'
    c_rf = '#3498db'
    
    # Plot curves starting at 1.0 (wealth index)
    ax.plot(dates_str, np.cumprod(1 + market_returns), label=f"Market EW (Cum: {m_market['Cum_Return']:.2%})", color=c_market, linestyle='--', linewidth=2)
    ax.plot(dates_str, np.cumprod(1 + ridge_ls), label=f"Ridge L-S Raw (Cum: {m_ridge['Cum_Return']:.1%})", color=c_ridge, linewidth=2.5)
    ax.plot(dates_str, np.cumprod(1 + fnn_ls), label=f"FNN L-S Raw (Cum: {m_fnn['Cum_Return']:.1%})", color=c_fnn, linewidth=2.5)
    ax.plot(dates_str, np.cumprod(1 + rf_ls), label=f"Random Forest L-S Raw (Cum: {m_rf['Cum_Return']:.1%})", color=c_rf, linewidth=2.5)
    
    # Styling
    ax.set_title('Figure 8: Wealth Index Out-of-Sample Backtest (No Frictions)\nStrategy: Long Top 10% vs. Short Bottom 10% (May 2025 - Apr 2026)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Date (Year-Month)', fontsize=11)
    ax.set_ylabel('Wealth Index (Growth of $1.00)', fontsize=11)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/backtest_performance_optimal_raw.png', dpi=300)
    plt.close()
    
    print("\nRaw backtest chart saved to 'plots/backtest_performance_optimal_raw.png'.")

if __name__ == '__main__':
    main()
