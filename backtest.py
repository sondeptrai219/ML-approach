import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import torch
import sys
import os

# Append project directory to path
sys.path.append("C:/Users/Asus/.gemini/antigravity/scratch/stock_fnn_predictor")
from model import StockPredictorFNN

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

def run_backtest(df_test, pred_col, target_col, pct=0.10):
    """
    Simulate a Long-Short strategy:
    For each month, go Long on top pct% predicted stocks, and Short on bottom pct% predicted stocks.
    """
    dates = sorted(df_test['Date'].unique())
    portfolio_returns = []
    
    for date in dates:
        month_data = df_test[df_test['Date'] == date].copy()
        n_stocks = len(month_data)
        k = max(1, int(n_stocks * pct)) # Number of stocks to select
        
        # Sort by predicted return
        month_sorted = month_data.sort_values(by=pred_col, ascending=False)
        
        # Long positions (top predicted returns)
        long_returns = month_sorted.head(k)[target_col].mean()
        
        # Short positions (bottom predicted returns)
        short_returns = month_sorted.tail(k)[target_col].mean()
        
        # Long-Short Return (Zero-cost net portfolio return)
        ls_return = long_returns - short_returns
        portfolio_returns.append(ls_return)
        
    return np.array(portfolio_returns)

def calculate_metrics(returns):
    """
    Calculate annualized return, volatility, Sharpe ratio, and Max Drawdown
    """
    cum_returns = np.cumprod(1 + returns) - 1
    ann_return = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Calculate Max Drawdown
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
    # 1. Preprocess and Split Data (Identical to training pipeline)
    print("Loading data...")
    df = pd.read_excel("data/fixed_dataset.xlsx")
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
    
    df_train = df_clean[df_clean['Date'].isin(unique_dates[:train_end_idx])]
    df_test = df_clean[df_clean['Date'].isin(unique_dates[val_end_idx:])].copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)
    y_train = df_train[target_col].values
    
    # 2. Get Predictions for all Models
    print("Generating predictions...")
    
    # Ridge
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_train, y_train)
    df_test['Pred_Ridge'] = ridge.predict(X_test)
    
    # RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    df_test['Pred_RF'] = rf.predict(X_test)
    
    # FNN
    fnn = StockPredictorFNN(len(feature_cols))
    fnn.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    fnn.eval()
    with torch.no_grad():
        df_test['Pred_FNN'] = fnn(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        
    # Benchmark: Equal-weighted Market Return (average return of all test stocks per month)
    market_returns = []
    dates = sorted(df_test['Date'].unique())
    for date in dates:
        market_returns.append(df_test[df_test['Date'] == date][target_col].mean())
    market_returns = np.array(market_returns)
    
    # 3. Run Backtest Simulations
    print("Running backtest simulations...")
    pct = 0.10  # Long top 10%, Short bottom 10%
    
    ridge_ls = run_backtest(df_test, 'Pred_Ridge', target_col, pct)
    rf_ls = run_backtest(df_test, 'Pred_RF', target_col, pct)
    fnn_ls = run_backtest(df_test, 'Pred_FNN', target_col, pct)
    
    # 4. Calculate performance metrics
    ridge_metrics = calculate_metrics(ridge_ls)
    rf_metrics = calculate_metrics(rf_ls)
    fnn_metrics = calculate_metrics(fnn_ls)
    market_metrics = calculate_metrics(market_returns)
    
    # Print results summary table
    print("\n================== BACKTEST METRICS (May 2025 - Apr 2026) ==================")
    metrics_data = [
        {'Strategy': 'Market Benchmark (Long)', **market_metrics},
        {'Strategy': 'Ridge Long-Short', **ridge_metrics},
        {'Strategy': 'FNN Long-Short', **fnn_metrics},
        {'Strategy': 'Random Forest (depth=3) L-S', **rf_metrics}
    ]
    metrics_df = pd.DataFrame(metrics_data)
    
    # Format percentages for cleaner printing
    print(metrics_df[[
        'Strategy', 'Cum_Return', 'Ann_Return', 'Ann_Vol', 'Sharpe', 'Max_DD'
    ]].to_string(index=False, formatters={
        'Cum_Return': lambda x: f"{x:.2%}",
        'Ann_Return': lambda x: f"{x:.2%}",
        'Ann_Vol': lambda x: f"{x:.2%}",
        'Sharpe': lambda x: f"{x:.2f}",
        'Max_DD': lambda x: f"{x:.2%}"
    }))
    print("==========================================================================")
    
    # Save CSV
    metrics_df.to_csv("results/backtest_metrics.csv", index=False)
    
    # 5. Plot Cumulative Return Wealth Curves
    plt.figure(figsize=(10, 6))
    
    dates_str = [d.strftime('%Y-%m') for d in dates]
    
    # Cumulative returns curves starting at 1.0 (wealth index)
    plt.plot(dates_str, np.cumprod(1 + market_returns), label=f"Market Equal-Weight (Cum: {market_metrics['Cum_Return']:.1%})", color='#7f8c8d', linestyle='--', linewidth=2)
    plt.plot(dates_str, np.cumprod(1 + ridge_ls), label=f"Ridge Long-Short (Cum: {ridge_metrics['Cum_Return']:.1%})", color='#9b59b6', linewidth=2.5)
    plt.plot(dates_str, np.cumprod(1 + fnn_ls), label=f"FNN Long-Short (Cum: {fnn_metrics['Cum_Return']:.1%})", color='#1abc9c', linewidth=2.5)
    plt.plot(dates_str, np.cumprod(1 + rf_ls), label=f"Random Forest L-S (Cum: {rf_metrics['Cum_Return']:.1%})", color='#3498db', linewidth=2.5)
    
    plt.title('Figure 6: Out-of-Sample Backtest Cumulative Returns\nStrategy: Long Top 10% vs. Short Bottom 10%', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date (Year-Month)', fontsize=12)
    plt.ylabel('Growth of $1.00 Investment', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/backtest_performance.png', dpi=300)
    plt.close()
    
    print("\nBacktest chart saved to 'plots/backtest_performance.png'.")

if __name__ == '__main__':
    main()
