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

def run_backtest_with_frictions(df_test, pred_col, target_col, pct=0.10, one_way_cost=0.0015, borrow_rate_annual=0.02):
    """
    Simulate Long-Short trading strategy accounting for transaction cost, slippage, and borrow costs.
    """
    dates = sorted(df_test['Date'].unique())
    portfolio_returns_raw = []
    portfolio_returns_net = []
    turnovers = []
    
    prev_long_tickers = set()
    prev_short_tickers = set()
    
    monthly_borrow_cost = borrow_rate_annual / 12.0
    
    for i, date in enumerate(dates):
        month_data = df_test[df_test['Date'] == date].copy()
        
        # Sort by predicted return
        month_sorted = month_data.sort_values(by=pred_col, ascending=False)
        n_stocks = len(month_sorted)
        k = max(1, int(n_stocks * pct))
        
        long_set = month_sorted.head(k)
        short_set = month_sorted.tail(k)
        
        long_tickers = set(long_set['ticker'].tolist())
        short_tickers = set(short_set['ticker'].tolist())
        
        long_ret = long_set[target_col].mean()
        short_ret = short_set[target_col].mean()
        
        # Raw LS return
        raw_ls_ret = long_ret - short_ret
        
        # Turnover calculation (proportion of portfolio changing)
        if i == 0:
            long_turnover = 1.0
            short_turnover = 1.0
        else:
            long_turnover = len(long_tickers - prev_long_tickers) / k
            short_turnover = len(short_tickers - prev_short_tickers) / k
            
        # Volume traded (volume required to buy/sell portfolio adjustments)
        # Building portfolio from scratch is 100% turnover (volume = 1.0)
        # Rebalancing requires selling leaving stock and buying new one, so volume is 2 * turnover.
        if i == 0:
            long_volume = 1.0
            short_volume = 1.0
        else:
            long_volume = 2.0 * long_turnover
            short_volume = 2.0 * short_turnover
            
        # Transaction costs
        tc_long = long_volume * one_way_cost
        tc_short = short_volume * one_way_cost
        total_tc = tc_long + tc_short
        
        # Net Return
        net_ls_ret = raw_ls_ret - total_tc - monthly_borrow_cost
        
        portfolio_returns_raw.append(raw_ls_ret)
        portfolio_returns_net.append(net_ls_ret)
        turnovers.append((long_turnover + short_turnover) / 2)
        
        prev_long_tickers = long_tickers
        prev_short_tickers = short_tickers
        
    return np.array(portfolio_returns_raw), np.array(portfolio_returns_net), np.mean(turnovers)

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
    df_test_full = df_clean[df_clean['Date'].isin(test_dates)].copy()
    
    print(f"Original Test data shape: {df_test_full.shape}")
    
    # 2. Apply Liquidity Filter (Exclude bottom 20% stocks by trading volume in each month)
    print("Applying 20% Liquidity Filter (volume_m)...")
    df_clean['Vol_Rank'] = df_clean.groupby('Date')['volume_m'].transform(lambda x: x.quantile(0.20))
    df_clean_filtered = df_clean[df_clean['volume_m'] >= df_clean['Vol_Rank']].copy()
    
    df_train_val_filtered = df_clean_filtered[df_clean_filtered['Date'].isin(train_dates)].copy()
    df_test_filtered = df_clean_filtered[df_clean_filtered['Date'].isin(test_dates)].copy()
    
    print(f"Filtered Train+Val shape: {df_train_val_filtered.shape}")
    print(f"Filtered Test shape: {df_test_filtered.shape} (Removed bottom 20% illiquid stocks)")
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(df_train_val_filtered[feature_cols].values)
    X_test = scaler.transform(df_test_filtered[feature_cols].values)
    y_train_val = df_train_val_filtered[target_col].values
    
    # 4. Fit Models on Filtered Train+Val Set
    print("\nTraining models on filtered Train+Val set...")
    
    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_train_val, y_train_val)
    df_test_filtered['Pred_Ridge'] = ridge.predict(X_test)
    
    # Random Forest (Optimal parameters from Split 4 validation search)
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_val, y_train_val)
    df_test_filtered['Pred_RF'] = rf.predict(X_test)
    
    # FNN (Optimal Split 4 configuration loaded from saved weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = len(feature_cols)
    fnn = DynamicFNN(input_dim, [64, 32, 16], 'LeakyReLU').to(device)
    
    if os.path.exists('best_fnn_Split_4.pth'):
        print("Loading optimal FNN weights from 'best_fnn_Split_4.pth'...")
        fnn.load_state_dict(torch.load('best_fnn_Split_4.pth', map_location=device))
    else:
        print("Warning: 'best_fnn_Split_4.pth' not found. Training FNN on filtered data for 15 epochs...")
        # Fallback train
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
        df_test_filtered['Pred_FNN'] = fnn(X_test_tensor).cpu().numpy().flatten()
        
    # Benchmark: Equal-weighted Market Return on filtered stocks
    market_returns = []
    dates = sorted(df_test_filtered['Date'].unique())
    for date in dates:
        market_returns.append(df_test_filtered[df_test_filtered['Date'] == date][target_col].mean())
    market_returns = np.array(market_returns)
    
    # 5. Run Backtests with frictions
    # Assumptions: 0.15% one-way cost (30 bps round-trip), 2.0% annual short borrow rate
    pct = 0.10
    one_way_cost = 0.0015
    borrow_rate = 0.02
    
    print("\nRunning portfolio simulations...")
    ridge_raw, ridge_net, ridge_turn = run_backtest_with_frictions(df_test_filtered, 'Pred_Ridge', target_col, pct, one_way_cost, borrow_rate)
    rf_raw, rf_net, rf_turn = run_backtest_with_frictions(df_test_filtered, 'Pred_RF', target_col, pct, one_way_cost, borrow_rate)
    fnn_raw, fnn_net, fnn_turn = run_backtest_with_frictions(df_test_filtered, 'Pred_FNN', target_col, pct, one_way_cost, borrow_rate)
    
    # Compute metrics
    m_market = calculate_metrics(market_returns)
    m_ridge_raw = calculate_metrics(ridge_raw)
    m_ridge_net = calculate_metrics(ridge_net)
    m_rf_raw = calculate_metrics(rf_raw)
    m_rf_net = calculate_metrics(rf_net)
    m_fnn_raw = calculate_metrics(fnn_raw)
    m_fnn_net = calculate_metrics(fnn_net)
    
    # Create comparison table
    backtest_results = [
        {'Strategy': 'Market Equal-Weight', 'Friction': 'No', 'Ann_Return': m_market['Ann_Return'], 'Ann_Vol': m_market['Ann_Vol'], 'Sharpe': m_market['Sharpe'], 'Max_DD': m_market['Max_DD'], 'Cum_Return': m_market['Cum_Return'], 'Avg_Turnover': 0.0},
        {'Strategy': 'Ridge Long-Short', 'Friction': 'No', 'Ann_Return': m_ridge_raw['Ann_Return'], 'Ann_Vol': m_ridge_raw['Ann_Vol'], 'Sharpe': m_ridge_raw['Sharpe'], 'Max_DD': m_ridge_raw['Max_DD'], 'Cum_Return': m_ridge_raw['Cum_Return'], 'Avg_Turnover': ridge_turn},
        {'Strategy': 'Ridge Long-Short', 'Friction': 'Yes', 'Ann_Return': m_ridge_net['Ann_Return'], 'Ann_Vol': m_ridge_net['Ann_Vol'], 'Sharpe': m_ridge_net['Sharpe'], 'Max_DD': m_ridge_net['Max_DD'], 'Cum_Return': m_ridge_net['Cum_Return'], 'Avg_Turnover': ridge_turn},
        {'Strategy': 'FNN Long-Short', 'Friction': 'No', 'Ann_Return': m_fnn_raw['Ann_Return'], 'Ann_Vol': m_fnn_raw['Ann_Vol'], 'Sharpe': m_fnn_raw['Sharpe'], 'Max_DD': m_fnn_raw['Max_DD'], 'Cum_Return': m_fnn_raw['Cum_Return'], 'Avg_Turnover': fnn_turn},
        {'Strategy': 'FNN Long-Short', 'Friction': 'Yes', 'Ann_Return': m_fnn_net['Ann_Return'], 'Ann_Vol': m_fnn_net['Ann_Vol'], 'Sharpe': m_fnn_net['Sharpe'], 'Max_DD': m_fnn_net['Max_DD'], 'Cum_Return': m_fnn_net['Cum_Return'], 'Avg_Turnover': fnn_turn},
        {'Strategy': 'Random Forest L-S', 'Friction': 'No', 'Ann_Return': m_rf_raw['Ann_Return'], 'Ann_Vol': m_rf_raw['Ann_Vol'], 'Sharpe': m_rf_raw['Sharpe'], 'Max_DD': m_rf_raw['Max_DD'], 'Cum_Return': m_rf_raw['Cum_Return'], 'Avg_Turnover': rf_turn},
        {'Strategy': 'Random Forest L-S', 'Friction': 'Yes', 'Ann_Return': m_rf_net['Ann_Return'], 'Ann_Vol': m_rf_net['Ann_Vol'], 'Sharpe': m_rf_net['Sharpe'], 'Max_DD': m_rf_net['Max_DD'], 'Cum_Return': m_rf_net['Cum_Return'], 'Avg_Turnover': rf_turn}
    ]
    
    results_df = pd.DataFrame(backtest_results)
    
    print("\n" + "="*50 + " BACKTEST PERFORMANCE SUMMARY WITH FRICTION " + "="*50)
    print(results_df.to_string(index=False, formatters={
        'Ann_Return': lambda x: f"{x:.2%}",
        'Ann_Vol': lambda x: f"{x:.2%}",
        'Sharpe': lambda x: f"{x:.2f}",
        'Max_DD': lambda x: f"{x:.2%}",
        'Cum_Return': lambda x: f"{x:.2%}",
        'Avg_Turnover': lambda x: f"{x:.2%}"
    }))
    print("="*145)
    
    # Save CSV
    results_df.to_csv("results/backtest_metrics_friction.csv", index=False)
    
    # 6. Plot curves net of frictions
    plt.figure(figsize=(10, 6))
    dates_str = [d.strftime('%Y-%m') for d in dates]
    
    plt.plot(dates_str, np.cumprod(1 + market_returns), label=f"Market EW (Cum: {m_market['Cum_Return']:.1%})", color='#7f8c8d', linestyle='--', linewidth=2)
    plt.plot(dates_str, np.cumprod(1 + ridge_net), label=f"Ridge L-S Net (Cum: {m_ridge_net['Cum_Return']:.1%})", color='#9b59b6', linewidth=2.5)
    plt.plot(dates_str, np.cumprod(1 + fnn_net), label=f"FNN L-S Net (Cum: {m_fnn_net['Cum_Return']:.1%})", color='#1abc9c', linewidth=2.5)
    plt.plot(dates_str, np.cumprod(1 + rf_net), label=f"Random Forest L-S Net (Cum: {m_rf_net['Cum_Return']:.1%})", color='#3498db', linewidth=2.5)
    
    plt.title('Figure 7: Wealth Index Net of Frictions & Liquidity Filtering\nFrictions: Slippage/Fee=15bps, Short Borrow=2% p.a., Liquidity Filter=Bottom 20%', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Date (Year-Month)', fontsize=12)
    plt.ylabel('Growth of $1.00 Investment', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/backtest_performance_friction.png', dpi=300)
    plt.close()
    
    print("\nFriction-adjusted backtest chart saved to 'plots/backtest_performance_friction.png'.")

if __name__ == '__main__':
    main()
