import torch
import pandas as pd
import numpy as np

from model import StockPredictorFNN

def main():
    # Initialize model and load weights
    # Input dimension is 37 features
    input_dim = 37
    model = StockPredictorFNN(input_dim)
    
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    state_dict = model.state_dict()
    
    # Write a detailed summary text file of all weights
    with open('results/fnn_weights_summary.txt', 'w') as f:
        f.write("==================================================\n")
        f.write("         FNN MODEL NODE WEIGHTS SUMMARY           \n")
        f.write("==================================================\n\n")
        
        for key, value in state_dict.items():
            f.write(f"Layer: {key}\n")
            f.write(f"Shape: {list(value.shape)}\n")
            # Calculate basic statistics for the weights in this layer
            weights_np = value.numpy()
            f.write(f"  Mean: {np.mean(weights_np):.6f}\n")
            f.write(f"  Std:  {np.std(weights_np):.6f}\n")
            f.write(f"  Min:  {np.min(weights_np):.6f}\n")
            f.write(f"  Max:  {np.max(weights_np):.6f}\n")
            f.write("-" * 50 + "\n\n")
            
    # Also save the raw weights to an Excel file with multiple sheets for inspection
    try:
        with pd.ExcelWriter('results/fnn_raw_weights.xlsx', engine='openpyxl') as writer:
            # Save Input BN parameters
            pd.DataFrame({
                'running_mean': state_dict['input_bn.running_mean'].numpy(),
                'running_var': state_dict['input_bn.running_var'].numpy(),
                'weight (gamma)': state_dict['input_bn.weight'].numpy(),
                'bias (beta)': state_dict['input_bn.bias'].numpy()
            }).to_excel(writer, sheet_name='Input_BatchNorm', index=False)
            
            # Save Layer 1 (64 x 37 weights)
            pd.DataFrame(state_dict['layer1.weight'].numpy()).to_excel(writer, sheet_name='Layer1_Weights', index=False)
            pd.DataFrame({'bias': state_dict['layer1.bias'].numpy()}).to_excel(writer, sheet_name='Layer1_Biases', index=False)
            
            # Save Layer 2 (32 x 64 weights)
            pd.DataFrame(state_dict['layer2.weight'].numpy()).to_excel(writer, sheet_name='Layer2_Weights', index=False)
            pd.DataFrame({'bias': state_dict['layer2.bias'].numpy()}).to_excel(writer, sheet_name='Layer2_Biases', index=False)
            
            # Save Layer 3 (16 x 32 weights)
            pd.DataFrame(state_dict['layer3.weight'].numpy()).to_excel(writer, sheet_name='Layer3_Weights', index=False)
            pd.DataFrame({'bias': state_dict['layer3.bias'].numpy()}).to_excel(writer, sheet_name='Layer3_Biases', index=False)
            
            # Save Output Layer (1 x 16 weights)
            pd.DataFrame(state_dict['output_layer.weight'].numpy()).to_excel(writer, sheet_name='Output_Weights', index=False)
            pd.DataFrame({'bias': state_dict['output_layer.bias'].numpy()}).to_excel(writer, sheet_name='Output_Biases', index=False)
            
        print("Raw weights exported successfully to 'results/fnn_raw_weights.xlsx'.")
    except Exception as e:
        print(f"Error exporting raw weights to Excel: {e}")
        
    # Print output layer weights for direct inspection
    print("\n--- Output Layer Weights (Weights connecting Hidden Layer 3 to the final output node) ---")
    output_weights = state_dict['output_layer.weight'].numpy()[0]
    output_bias = state_dict['output_layer.bias'].numpy()[0]
    
    for i, w in enumerate(output_weights):
        print(f"Node {i:02d} weight: {w:+.6f}")
    print(f"Output Bias:      {output_bias:+.6f}")

if __name__ == '__main__':
    main()
