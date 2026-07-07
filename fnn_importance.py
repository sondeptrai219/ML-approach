import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Import dataset and model structure
from model import StockDataset, StockPredictorFNN

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

def main():
    # 1. Recreate dataset pipeline
    file_path = "data/fixed_dataset.xlsx"
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    id_cols = ['ticker', 'Date', 'mapped_quarter']
    target_col = 'Target'
    feature_cols = [col for col in df.columns if col not in id_cols and col != target_col]
    
    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'ticker']).reset_index(drop=True)
    
    # Split (Identical Train/Test Split)
    unique_dates = sorted(df_clean['Date'].unique())
    train_end_idx = int(0.75 * len(unique_dates))
    val_end_idx = int(0.875 * len(unique_dates))
    
    train_dates = unique_dates[:train_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    df_train = df_clean[df_clean['Date'].isin(train_dates)]
    df_test = df_clean[df_clean['Date'].isin(test_dates)]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)
    y_test = df_test[target_col].values
    
    # 2. Load trained FNN model
    input_dim = len(feature_cols)
    model = StockPredictorFNN(input_dim)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. Baseline Prediction
    print("Calculating FNN baseline predictions...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        baseline_preds = model(X_test_tensor).cpu().numpy().flatten()
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    print(f"Baseline Test MSE: {baseline_mse:.6f}")
    
    # 4. Permutation Importance Loop
    importances = []
    for i, col in enumerate(feature_cols):
        # Create a copy of the test set
        X_test_permuted = X_test.copy()
        
        # Shuffle only column i
        np.random.shuffle(X_test_permuted[:, i])
        
        # Predict on permuted data
        X_permuted_tensor = torch.tensor(X_test_permuted, dtype=torch.float32).to(device)
        with torch.no_grad():
            permuted_preds = model(X_permuted_tensor).cpu().numpy().flatten()
            
        permuted_mse = mean_squared_error(y_test, permuted_preds)
        
        # Importance = increase in MSE (how much worse the model gets)
        importance = permuted_mse - baseline_mse
        importances.append(importance)
        
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 15 FNN Permutation Importances ---")
    print(importance_df.head(15).to_string(index=False))
    
    # Save CSV
    importance_df.to_csv("results/fnn_importances.csv", index=False)
    
    # 5. Plotting
    top_n = 15
    top_df = importance_df.head(top_n).iloc[::-1]  # Reverse for ascending order in horizontal bar plot
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_df['Feature'], top_df['Importance'], color='#1abc9c', edgecolor='none', height=0.6)
    
    # Customize grid and lines
    plt.axvline(x=0, color='#7f8c8d', linestyle='-', linewidth=0.8)
    plt.title(f'FNN Feature Importance (Permutation on Test Set)\nMetric: Increase in Mean Squared Error (MSE)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance (Permuted MSE - Baseline MSE)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    # Add values to the right of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(top_df['Importance']) * 0.01), bar.get_y() + bar.get_height()/2, 
                 f'+{width:.6f}', 
                 va='center', ha='left', fontsize=9, color='#2c3e50', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('plots/fnn_feature_importances.png', dpi=300)
    plt.close()
    
    print("\nFNN feature importance chart saved to 'plots/fnn_feature_importances.png'.")

if __name__ == '__main__':
    main()
