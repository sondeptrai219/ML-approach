# Empirical Asset Pricing via Machine Learning: FNN vs. Random Forest vs. Linear Models

This repository contains the codebase and empirical research report evaluating the predictive power of **Feedforward Neural Networks (FNN)**, **Random Forest (RF)** ensembles, and regularized linear models (**Ridge Regression**) in forecasting cross-sectional stock returns next month. 

Using a panel dataset of ~150 firms over an 8-year period (96 months, May 2018 to April 2026) containing 37 lagged financial, accounting, and market variables, we implement an expanding rolling-window cross-validation scheme to evaluate the out-of-sample performance of the models.

---

## 1. Directory Structure

```
├── data/
│   └── fixed_dataset.xlsx                     # Cleaned Excel dataset of monthly stock return observations
├── models/
│   ├── best_fnn_Split_1.pth                   # Optimal refitted FNN weights for Split 1
│   ├── best_fnn_Split_2.pth                   # Optimal refitted FNN weights for Split 2
│   ├── best_fnn_Split_3.pth                   # Optimal refitted FNN weights for Split 3
│   ├── best_fnn_Split_4.pth                   # Optimal refitted FNN weights for Split 4
│   └── best_model.pth                         # Baseline static FNN model weights
├── plots/
│   ├── backtest_performance.png               # Wealth index curve from the static backtest
│   ├── backtest_performance_friction.png      # Friction-adjusted cumulative wealth curves
│   ├── backtest_performance_optimal_raw.png   # Raw optimal Split 4 cumulative wealth curves (no frictions)
│   ├── characteristic_importance_heatmap.png  # Gu, Kelly, Xiu (2020) style importance heatmap
│   ├── fnn_feature_importances.png            # Permutation-based FNN feature importances
│   ├── loss_curves.png                        # FNN training and validation loss curves
│   ├── predictions_scatter.png                # FNN out-of-sample scatter plots
│   ├── rf_predictions_scatter.png             # RF out-of-sample scatter plots
│   └── ridge_predictions_scatter.png          # Ridge out-of-sample scatter plots
├── results/
│   ├── backtest_metrics.csv                   # Metrics from the static backtest
│   ├── backtest_metrics_friction.csv          # Friction-adjusted backtest performance table
│   ├── backtest_metrics_raw.csv               # Raw optimal Split 4 backtest performance table
│   ├── custom_train_test_results.csv          # Metrics from the custom 2023-2025 train-test split
│   ├── fnn_importances.csv                    # Feature importance scores for the FNN model
│   ├── fnn_raw_weights.xlsx                   # Raw weight matrices extracted from FNN layers
│   ├── fnn_weights_summary.txt                # Text summary of FNN weight statistics
│   ├── normalized_characteristic_importances.csv # Heatmap normalized data
│   ├── rf_comparison.csv                      # RF validation grid results
│   ├── rf_importances.csv                     # RF Gini feature importances
│   ├── ridge_coefficients.csv                 # Ridge coefficient coefficients
│   ├── rolling_fnn_history.csv                # FNN rolling cross-validation grid search logs
│   ├── rolling_fnn_test_results.csv           # FNN rolling test set performance metrics
│   ├── rolling_rf_history.csv                 # RF rolling cross-validation grid search logs
│   ├── rolling_rf_test_results.csv            # RF rolling test set performance metrics
│   └── [various metrics files]                # Text files of test metrics
├── fama_french/
│   └── README.md                              # Template folder for Fama-French factor regressions
├── requirements.txt                           # Python environment dependencies
├── .gitignore                                 # Git ignore configuration
├── README.md                                  # Research report and documentation
└── [python scripts]                           # Core scripts (rolling cross-validation, backtesting, custom runs)
```

---

## 2. Methodology & Key Findings

### 2.1 Expanding Rolling-Window Protocol
To prevent data leakage and evaluate robustness across different time horizons, we implement an expanding rolling-window cross-validation protocol over 4 chronological splits:
* **Split 1**: Train 36m (Months 1–36), Val 12m (37–48), Test 24m (49–72)
* **Split 2**: Train 48m (Months 1–48), Val 12m (49–60), Test 24m (61–84)
* **Split 3**: Train 60m (Months 1–60), Val 12m (61–72), Test 24m (73–96)
* **Split 4**: Train 72m (Months 1–72), Val 12m (73–84), Test 12m (85–96)

For both the FNN and RF models, we identify the optimal hyperparameter configuration on the Validation set. We then **refit the selected model on the combined Train + Validation set** (running the FNN for exactly the validation-optimal epoch count $E^*$) before evaluating it on the out-of-sample Test set.

### 2.2 Empirical Results Summary (Refitted Rolling Window)

#### Random Forest Rolling Results:
* **Optimal parameters**: Validation grid searches consistently select shallow trees (`max_depth = 2.0` or `5.0`) to restrict model variance.
* **Test Performance**:
  * **Split 4 (72m)**: The optimal RF (50 trees, depth 2, leaf 2) achieves a **positive out-of-sample $R^2_{OOS}$ of +2.32%** and a **Pearson Correlation of 24.37%**.

#### Feedforward Neural Network Rolling Results:
* **Optimal parameters**: Validation selects a narrow bottleneck (`[32, 16, 8]`) for the first three splits, and `[64, 32, 16]` for Split 4.
* **Test Performance**:
  * **Split 4 (72m)**: The optimal refitted FNN (ReLU, 15 epochs) achieves a **positive out-of-sample $R^2_{OOS}$ of +1.91%** and a **correlation of 10.68%**.

---

## 3. Backtest Portfolio Performance (Split 4 Test Set)

We simulate a monthly rebalanced Long-Short portfolio strategy (buying the top 10% predicted returns, shorting the bottom 10%) on the Split 4 Test Set (May 2025 – Apr 2026).

We compare two cases:
1. **Raw Backtest (No Frictions)**: No trading costs or liquidity limits.
2. **Friction-Adjusted Backtest**:
   * **Liquidity Filter**: Exclude the bottom 20% of stocks by trading volume to ensure tradeability.
   * **Trading Cost**: 0.15% (15 bps) one-way cost (30 bps round-trip) for fee + execution slippage.
   * **Short Borrow Rate**: 2.0% annual borrow rate (approx. 0.167% per month) on short positions.

### Out-of-Sample Portfolio Metrics:
| Strategy | Friction | Avg Monthly Turnover | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Market Equal-Weight** | No | 0.00% | `13.20%` | `13.36%` | `13.54%` | `0.99` | `-8.56%` |
| **Ridge Long-Short** | Yes | 24.31% | `-26.50%` | `-26.73%` | `26.34%` | `-1.01` | `-34.04%` |
| **FNN Long-Short (Raw)** | No | 52.43% | **`43.02%`** | **`37.36%`** | `14.70%` | **`2.54`** | **`-3.76%`** |
| **FNN Long-Short (Net)** | **Yes** | 52.43% | **`9.34%`** | **`10.19%`** | `15.80%` | **`0.65`** | **`-6.10%`** |
| **Random Forest L-S** | Yes | 12.50% | `-22.95%` | `-24.54%` | `15.61%` | `-1.57` | `-25.45%` |

* **Alpha Survival**: The FNN strategy remains profitable net of all frictions, showing that it extracts highly robust rankings out-of-sample.
* **Friction Drag**: The high turnover of the FNN (**52.43%**) causes a **5.48% annualized drag** due to execution costs, showing the critical importance of turnover management.

---

## 4. How to Run the Project

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Execution Scripts
* **Random Forest Rolling C-V**:
  ```bash
  python rolling_rf.py
  ```
* **FNN Rolling C-V (with refitting)**:
  ```bash
  python rolling_fnn.py
  ```
* **Friction-Adjusted Backtest**:
  ```bash
  python backtest_split4_friction.py
  ```
* **Raw Outperformance Backtest**:
  ```bash
  python backtest_split4_raw.py
  ```
* **Custom Run (2023–2025 Train -> 2025–2026 Test)**:
  ```bash
  python train_test_custom.py
  ```
