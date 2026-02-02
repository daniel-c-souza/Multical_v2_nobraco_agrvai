import numpy as np
import os
import sys
import ast
import joblib

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.multical.models.svr import SVRModel
from src.multical.preprocessing.pipeline import apply_pretreatment

def load_training_data(data_files):
    x_list = []
    absor_list = []
    wavelengths = None
    
    print("Loading data...")
    for x_f, abs_f in data_files:
        if os.path.exists(x_f) and os.path.exists(abs_f):
            try:
                xi = np.loadtxt(x_f)
                with open(abs_f, 'r') as f_node:
                    header_str = f_node.readline().strip()
                header_parts = header_str.split()
                if header_parts[0].lower() == 'time':
                    wl_curr = np.array([float(x) for x in header_parts[1:]])
                else:
                    wl_curr = np.array([float(x) for x in header_parts])
                absi = np.loadtxt(abs_f, skiprows=1)
                
                if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:]
                elif xi.ndim == 1: xi = xi.reshape(-1, 1)
                if absi.ndim == 1: absi = absi.reshape(1, -1)
                data_curr = absi[:, 1:] 
                
                if wavelengths is None: wavelengths = wl_curr
                x_list.append(xi)
                absor_list.append(data_curr)
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")
    
    if not x_list: return None, None, None
    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    absor0 = np.vstack([wavelengths, absor_data])
    return x0, absor0, wavelengths

def get_best_params(results_dir, nc):
    minimos_path = os.path.join(results_dir, 'minimos.txt')
    log_path = os.path.join(results_dir, 'params_log.txt')
    
    if not os.path.exists(minimos_path) or not os.path.exists(log_path):
        print("Results not found. Run Grid Search first.")
        return None

    # Read minimos
    # Row 0: RMSE, Row 1: Index (1-based)
    min_data = np.loadtxt(minimos_path)
    best_indices = min_data[1, :].astype(int)
    
    # Read params log
    params_map = {}
    with open(log_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx = int(parts[0])
                p_dict = ast.literal_eval(parts[1])
                params_map[idx] = p_dict
    
    best_params_list = []
    for j in range(nc):
        idx = best_indices[j]
        if idx in params_map:
            best_params_list.append(params_map[idx])
        else:
            print(f"Warning: Index {idx} not found in log.")
            best_params_list.append({'C':1, 'epsilon':0.1, 'gamma':'scale', 'kernel':'rbf'}) # Default
            
    return best_params_list

def main():
    results_dir = "results_svr_experiment"
    cname = ['cb', 'gl', 'xy']
    nc = len(cname)
    
    # Files
    data_files = [ 
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),
        ('exp7_refe.txt', 'exp7_nonda.txt'),
    ]
    
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 2, 1, 1],
    ]
    
    # 1. Load Data
    Y_train_all, X_train_raw_with_wl, wavelengths = load_training_data(data_files)
    if Y_train_all is None: return

    lambda0 = X_train_raw_with_wl[0, :]
    X_train_raw = X_train_raw_with_wl[1:, :]
    
    # 2. Preprocess X
    print("Applying Pretreatment...")
    X_train_proc, _ = apply_pretreatment(pretreat, X_train_raw, lambda0)
    
    # 3. Get Best Params
    print("Retrieving best parameters...")
    best_params_list = get_best_params(results_dir, nc)
    if not best_params_list: return

    # 4. Train and Save per Component
    for j in range(nc):
        print(f"\n--- Training SVR for {cname[j]} ---")
        bp = best_params_list[j]
        print(f"Params: {bp}")
        
        # Prepare Target (Single Column)
        y_target = Y_train_all[:, j:j+1]
        
        # Normalize Target (Important for SVR)
        # We handle this normalization manually here or inside SVRModel?
        # SVRModel uses StandardScaler on Y internally if we pass it.
        # But wait, in svr_engine we passed scaled Y to fit_predict?
        # In svr_engine.py:
        #   x_max = max(x)
        #   x_norm = x / x_max
        #   y_target_col = x_cal[:, j]  <-- This was normalized!
        #
        # SVRModel.fit_predict internaly does `StandardScaler`.
        # Double scaling?
        # In engine: We manually normalized by Max.
        # Inside SVRModel: It standardizes (Mean/Std).
        # This is fine, but we must replicate the Manual Normalization if we want the result to match the grid search conditions?
        # Actually, standardizing is usually better than Max normalization for SVR.
        # The engine did Max normalization to compute RMSE in normalized units roughly?
        # Let's trust SVRModel's internal StandardScaler.
        # So we pass raw Y_target (g/L) to SVRModel?
        # If we pass raw Y, SVRModel scales it.
        # BUT, if `svr_engine` used `x_norm` (Max scaled) as input to `fit_predict`, then the optimal parameters (C, epsilon) are optimized for that scale (0-1 range).
        # If we now pass raw data (0-50 range), the optimal 'epsilon' (e.g. 0.001) might be too small!
        # C is less sensitive to scale shift if X is scaled, but epsilon is directly related to Y scale.
        # If epsilon=0.01 was best for Y in [0,1], then for Y in [0, 50], epsilon should probably be 0.5.
        
        # CRITICAL DECISION:
        # The engine trained on `x_norm` (Max Scaled).
        # So the hyperparameters are valid for Max Scaled data.
        # Therefore, we MUST scale Y by the same max factors before training!
        # AND we must stick to that scaling during inference.
        
        # Calculate Y Max
        y_max = np.max(Y_train_all[:, j])
        if y_max == 0: y_max = 1.0
        
        y_train_norm = y_target / y_max
        
        # Train
        model = SVRModel()
        # We pass X_train_proc and y_train_norm.
        # SVRModel will internally do StandardScaler on top of this. That's fine.
        # X_test is same as X_train for final fit (we don't care about prediction here, just fit)
        model.fit_predict(X_train_proc, y_train_norm, X_train_proc, 
                          C=bp['C'], epsilon=bp['epsilon'], gamma=bp['gamma'], kernel=bp['kernel'])
        
        # Save Model State
        # We also need to save 'y_max' to un-scale predictions later!
        # SVRModel doesn't know about our external max-scaling.
        # I will hack: Add `custom_y_scale` attribute to the model object before saving.
        model.custom_y_scale = y_max
        
        save_path = os.path.join(results_dir, f'svr_model_{cname[j]}.pkl')
        model.save_model(save_path)
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
