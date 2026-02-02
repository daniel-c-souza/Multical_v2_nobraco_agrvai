import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define the grid exactly as it was in ann_engine.py for the overnight run
def get_param_from_index(k_idx):
    # k_idx is 0-based index
    # Matched from ann_engine.py
    grid_hidden = [5, 10, 15, 25, 50, 100, 200, 400, 500]
    grid_layers = [1, 2, 3, 4, 5, 6, 7, 8]
    grid_act = ['relu', 'tanh', 'sigmoid', 'silu', 'leaky_relu']
    grid_lr = [0.005, 0.001, 0.0001]

    param_list = []
    for h in grid_hidden:
        for lay in grid_layers:
            for act in grid_act:
                for lr in grid_lr:
                    param_list.append({'hidden': h, 'layers': lay, 'act': act, 'lr': lr})
    
    if k_idx < len(param_list):
        return param_list[k_idx]
    else:
        return None

def main():
    results_dir = "results_ann_experiment"
    
    # Load Data
    erro_cv_path = os.path.join(results_dir, "Erro_cv.txt")
    erro_cal_path = os.path.join(results_dir, "Erro_cal.txt")
    
    if not os.path.exists(erro_cv_path):
        print(f"File not found: {erro_cv_path}")
        return

    # Load RMSE Arrays
    # Format: k, col1, col2, col3 ...
    try:
        data_cv = np.loadtxt(erro_cv_path, skiprows=1) # Skip header
        data_cal = np.loadtxt(erro_cal_path, skiprows=1)
        
        with open(erro_cv_path, 'r') as f:
            header = f.readline().strip().split('\t')
            cnames = header[1:] # Skip 'k'
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    k_col = data_cv[:, 0]
    rmse_cv = data_cv[:, 1:]
    rmse_cal = data_cal[:, 1:]
    
    nc = len(cnames)
    
    # Find Minimums
    min_indices = np.argmin(rmse_cv, axis=0)
    min_vals = np.min(rmse_cv, axis=0)
    
    print("\n" + "="*60)
    print(f"{'Component':<10} | {'RMSECV':<10} | {'Index':<6} | {'Parameters'}")
    print("-" * 60)
    
    for j in range(nc):
        idx = min_indices[j]
        val = min_vals[j]
        k_val = int(k_col[idx])
        
        # Map back to 0-based index for param list
        params = get_param_from_index(k_val - 1) 
        
        param_str = "Unknown"
        if params:
            param_str = f"H={params['hidden']}, L={params['layers']}, Act={params['act']}, LR={params['lr']}"
            
        print(f"{cnames[j]:<10} | {val:.5f}    | {k_val:<6} | {param_str}")
        
    print("="*60 + "\n")

    # Plotting
    fig, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
    axes = axes.flatten()
    
    colors = ['green', 'red', 'purple', 'blue', 'orange']
    
    for j in range(nc):
        ax = axes[j]
        c_curr = colors[j % len(colors)]
        
        ax.plot(k_col, rmse_cv[:, j], color=c_curr, linestyle='-', label='RMSECV')
        ax.plot(k_col, rmse_cal[:, j], color=c_curr, linestyle='--', alpha=0.6, label='RMSEC')
        
        best_k_idx = min_indices[j]
        best_k = k_col[best_k_idx]
        min_r = min_vals[j]
        
        ax.plot(best_k, min_r, 'k*', markersize=15, label=f'Min (k={int(best_k)})')
        
        ax.set_xlabel('Grid Index')
        ax.set_ylabel(f'RMSE ({cnames[j]})')
        ax.set_title(f'{cnames[j]} Optimization w/ ANN')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    main()
