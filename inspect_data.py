import pandas as pd
import numpy as np
import os

file_path = "data/fixed_dataset.xlsx"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit(1)

print("Loading dataset...")
try:
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist()[:10], "... (total", len(df.columns), "columns)")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nData description:")
    print(df.info())
    print("\nChecking missing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
except Exception as e:
    print(f"Error loading file: {e}")
