import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import joblib

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.multical.models.svr import SVRModel
from src.multical.preprocessing.pipeline import apply_pretreatment

def load_rmsecv(results_dir, nc):
    path = os.path.join(results_dir, 'minimos.txt')
    if not os.path.exists(path):
        return np.zeros(nc)
    data = np.loadtxt(path)
    return data[0, :] # Row 0 is RMSE_conc

def main():
    results_dir = "results_svr_experiment"
    cname = ['cb', 'gl', 'xy']
    nc = len(cname)
    colors = ['green', 'red', 'purple']
    
    # 1. Load Validation/Error Stats
    rmse_vals = load_rmsecv(results_dir, nc)
    print(f"Loaded RMSECV: {rmse_vals}")

    # 2. Load Models
    models = []
    for j in range(nc):
        path = os.path.join(results_dir, f'svr_model_{cname[j]}.pkl')
        if not os.path.exists(path):
            print(f"Model file not found: {path} - Run training first.")
            return
        
        # Load SVRModel
        model = SVRModel()
        model.load_model(path)
        
        # Check for custom scale
        if not hasattr(model, 'custom_y_scale'):
            print(f"Warning: model {cname[j]} missing custom_y_scale. Assuming 1.0")
            model.custom_y_scale = 1.0
            
        models.append(model)

    # 3. Pretreatment Settings (Must Match Training)
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 2, 1, 1],
    ]

    # 4. Find Inference Files
    # Assuming files are in project root (workspace root)
    # Filter for *_inf.txt
    workspace_root = project_root 
    all_files = os.listdir(workspace_root)
    inf_files = [f for f in all_files if f.endswith('_inf.txt')]
    
    if not inf_files:
        print("No *_inf.txt files found in workspace root.")
        return

    print(f"Found {len(inf_files)} inference files: {inf_files}")

    # 5. Process Each File
    for f_name in inf_files:
        f_path = os.path.join(workspace_root, f_name)
        print(f"\nProcessing {f_name}...")
        
        # Determine Reference File
        # Map: exp_04_inf.txt -> exp4_refe.txt
        ref_file = None
        if 'exp_04' in f_name or 'exp4' in f_name: ref_file = 'exp4_refe.txt'
        elif 'exp_05' in f_name or 'exp5' in f_name: ref_file = 'exp5_refe.txt'
        elif 'exp_06' in f_name or 'exp6' in f_name: ref_file = 'exp6_refe.txt'
        elif 'exp_07' in f_name or 'exp7' in f_name: ref_file = 'exp7_refe.txt'

        ref_times = None
        ref_values = None
        
        if ref_file:
             ref_path = os.path.join(workspace_root, ref_file)
             if os.path.exists(ref_path):
                 print(f"  Loading reference data from: {ref_file}")
                 try:
                     ref_data = np.loadtxt(ref_path)
                     if ref_data.ndim > 1 and ref_data.shape[1] >= 4:
                         ref_times = ref_data[:, 0] * 60 # Convert hours to minutes
                         ref_values = ref_data[:, 1:4] # Cols 1,2,3 for Cb,Gl,Xy

                         # Handle Duplicates by Averaging
                         if len(ref_times) != len(np.unique(ref_times)):
                             print(f"  Note: Duplicate time points found in {ref_file}. Averaging values.")
                             unique_times = np.unique(ref_times)
                             new_values = []
                             for t in unique_times:
                                 indices = np.where(ref_times == t)[0]
                                 new_values.append(np.mean(ref_values[indices], axis=0))
                             ref_times = unique_times
                             ref_values = np.array(new_values)

                 except Exception as e:
                     print(f"  Warning: Failed to load reference {ref_file}: {e}")

        try:
            # Read Header for Wavelengths
            with open(f_path, 'r') as f:
                header_parts = f.readline().strip().split()
            
            # Use 'Time' detection or assume first col is time
            if header_parts[0].lower() == 'time':
                lambda_raw = np.array([float(x) for x in header_parts[1:]])
                start_col_data = 1
            else:
                # If no 'Time' label, assume format similar to training
                lambda_raw = np.array([float(x) for x in header_parts])
                start_col_data = 1 # Still assume col 0 is time data
            
            # Read Data
            data_full = np.loadtxt(f_path, skiprows=1)
            if data_full.ndim == 1: data_full = data_full.reshape(1, -1)
            
            times = data_full[:, 0]
            spectra = data_full[:, 1:]
            
            # Preprocess
            spectra_proc, _ = apply_pretreatment(pretreat, spectra, lambda_raw)
            
            # Predict
            fig, axs = plt.subplots(nc, 1, figsize=(10, 8), sharex=True)
            if nc == 1: axs = [axs]
            
            for j in range(nc):
                model = models[j]
                # Predict
                # SVRModel expects X_test as 3rd arg to fit_predict to return prediction
                # BUT we need a pure 'predict' method.
                # Oh, I updated SVRModel to have 'predict(X)' method? 
                # Let's check svr.py content...
                # I see 'save_model', 'load_model'. 
                # Did I add 'predict'? 
                # I must verify. If not, I can use fit_predict with dummy? No, that fits again!
                # I need 'predict'.
                
                # Checking memory of svr.py edit... 
                # I remember reading it, but did I add predict? 
                # fit_predict fits scalars. If I load scalars, I need to apply them transform -> sklearn predict -> inverse transform.
                # The 'load_model' loads the whole 'self' dictionary.
                # If SVRModel class doesn't have a 'predict' method, I cannot infer without fitting.
                # I MUST ADD 'predict' to SVRModel in src/multical/models/svr.py if it's missing.
                
                # Assuming I add it or it exists. Let's write the code assuming it exists.
                # If it doesn't, I will fix svr.py next.
                
                # Wait, if SVRModel is just a wrapper, maybe the pickle saves the wrapper instance?
                # Joblib saves the 'dict' of the object usually? Or the object itself?
                # def save_model(self, path): joblib.dump(self.__dict__, path)
                # def load_model(self, path): self.__dict__.update(joblib.load(path))
                
                # So if I add a method to the CLASS now, the loaded OBJECT (which is just state restoration) will have access to the new class method.
                # So I just need to ensure the class has `predict`.
                
                y_pred_norm = model.predict(spectra_proc) 
                
                # MultiOutput returns (N, 1) or (N,).
                y_pred_norm = y_pred_norm.flatten()
                
                # Apply Custom Scale (since we trained on y/y_max)
                y_pred_real = y_pred_norm * model.custom_y_scale
                
                # Plot
                ax = axs[j]
                ax.plot(times, y_pred_real, label='Predicted', color=colors[j])
                
                # Add Error Band (RMSECV)
                rmse = rmse_vals[j]
                ax.fill_between(times, y_pred_real - rmse, y_pred_real + rmse, 
                                color=colors[j], alpha=0.2, label=f'RMSECV ({rmse:.2f})')
                
                # Add Reference Scatter
                if ref_times is not None and ref_values is not None:
                    ax.scatter(ref_times, ref_values[:, j], color=colors[j], edgecolors='black', marker='o', s=40, zorder=5, label='Offline Reference')
                
                ax.set_ylabel(f'{cname[j]} (g/L)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.5)
            
            axs[-1].set_xlabel('Time (min)') 
            plt.suptitle(f'SVR Inference: {f_name}')
            plt.tight_layout()
            plt.show() # Interactive
            
        except Exception as e:
            print(f"Failed to process {f_name}: {e}")

if __name__ == "__main__":
    main()
