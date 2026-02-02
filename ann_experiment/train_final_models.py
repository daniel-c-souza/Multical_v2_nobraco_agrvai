import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.multical.models.ann_torch import ANNModel
from src.multical.preprocessing.pipeline import apply_pretreatment

def main():
    print("=== Training Final ANN Models ===")
    
    # 1. Configuration
    models_dir = os.path.join(project_root, "models", "ann_best")
    os.makedirs(models_dir, exist_ok=True)
    
    # Best Parameters from Overnight Run
    # cb: H=500, L=3, Act=sigmoid, LR=0.0001
    # gl: H=25, L=1, Act=sigmoid, LR=0.001
    # xy: H=25, L=1, Act=sigmoid, LR=0.001
    
    best_params = {
        'cb': {'hidden': 200, 'layers': 5, 'act': 'leaky_relu', 'lr': 0.0001},
        'gl': {'hidden': 50, 'layers': 2, 'act': 'sigmoid', 'lr': 0.001},
        'xy': {'hidden': 10, 'layers': 1, 'act': 'leaky_relu', 'lr': 0.001}
    }
    
    cnames = ['cb', 'gl', 'xy']
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 1, 2, 1, 1],
    ]
    
    # 2. Load Data
    data_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),
        ('exp7_refe.txt', 'exp7_nonda.txt')
    ]
    
    x_list = []
    absor_list = []
    wavelengths = None
    
    print("Loading training data...")
    for x_f, abs_f in data_files:
        path_x = os.path.join(project_root, x_f)
        path_abs = os.path.join(project_root, abs_f)
        
        if os.path.exists(path_x) and os.path.exists(path_abs):
            try:
                # Load X (Reference)
                xi = np.loadtxt(path_x)
                if xi.ndim == 2 and xi.shape[1] > 1:
                    xi = xi[:, 1:] 
                
                # Load Absorbance
                with open(path_abs, 'r') as f_node:
                    header_parts = f_node.readline().strip().split()
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                absi = np.loadtxt(path_abs, skiprows=1)
                data_curr = absi[:, 1:]
                
                if wavelengths is None:
                    wavelengths = wl_curr
                
                x_list.append(xi)
                absor_list.append(data_curr)
            except Exception as e:
                print(f"Error loading {x_f}: {e}")
                
    if not x_list:
        print("No data loaded.")
        return

    x_cal = np.vstack(x_list)
    absor0 = np.vstack(absor_list)
    
    # 3. Pretreatment
    print("Applying Pretreatment...")
    absor, wl_new = apply_pretreatment(pretreat, absor0, wavelengths, plot=False)
    
    # 4. Train Individual Models (Single-Output Training)
    # matching the Single-Output Grid Search architecture.
    for i, name in enumerate(cnames):
        print(f"\n--- Training Model Optimized for {name} ---")
        params = best_params[name]
        print(f"Params: {params}")
        
        # Select single column for Y
        y_single = x_cal[:, [i]]
        
        model = ANNModel()
        
        model.fit_predict(
            X_train=absor, 
            Y_train=y_single, 
            X_test=absor, 
            hidden_units=params['hidden'], 
            n_layers=params['layers'], 
            activation=params['act'], 
            learning_rate=params['lr'], 
            epochs=500,            
            early_stopping=False   
        )
        
        # Save
        save_path = os.path.join(models_dir, f"model_{name}")
        model.save_model(save_path)
        
    print("\nAll models trained and saved successfully.")

if __name__ == "__main__":
    main()
