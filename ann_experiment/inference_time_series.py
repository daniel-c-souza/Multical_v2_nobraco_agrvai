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
    print("=== ANN Time Series Inference ===")
    
    # 1. Configuration
    models_dir = os.path.join(project_root, "models", "ann_best")
    output_dir = os.path.join(project_root, "results_inference_ann", "time_series")
    os.makedirs(output_dir, exist_ok=True)
    
    cnames = ['cb', 'gl', 'xy']
    colors = ['green', 'red', 'purple']
    
    # Must match training exactly!
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 1, 2, 1, 1],
    ]
    
    # Files to predict
    inf_files = [
        'exp_04_inf.txt',
        'exp_05_inf.txt',
        'exp_06_inf.txt'
    ]
    
    # 2. Load Models
    print("Loading Models...")
    models = {}
    for name in cnames:
        model_path = os.path.join(models_dir, f"model_{name}")
        model = ANNModel()
        try:
            model.load_model(model_path)
            models[name] = model
            print(f"  Loaded model_{name}")
        except Exception as e:
            print(f"  Failed to load model for {name}: {e}")
            return

    # Load RMSECV for error bars
    rmse_path = os.path.join(project_root, "results_ann_experiment", "minimos.txt")
    rmsecv_vals = [0.0, 0.0, 0.0]
    if os.path.exists(rmse_path):
        try:
             # row 0 is RMSECV, row 1 is index
             min_data = np.loadtxt(rmse_path)
             if min_data.ndim == 2:
                  rmsecv_vals = min_data[0, :]
             print(f"  Loaded RMSECV: {rmsecv_vals}")
        except Exception as e:
             print(f"  Could not load RMSECV from minimos.txt: {e}")
    else:
        print("  Warning: results_ann_experiment/minimos.txt not found. Error bands will be 0.")

    # 3. Process Each File
    for filename in inf_files:
        print(f"\nProcessing {filename}...")
        
        # Load Reference Data if available
        # Map: exp_04_inf.txt -> exp4_refe.txt
        ref_file = None
        if 'exp_04' in filename or 'exp4' in filename: ref_file = 'exp4_refe.txt'
        elif 'exp_05' in filename or 'exp5' in filename: ref_file = 'exp5_refe.txt'
        elif 'exp_06' in filename or 'exp6' in filename: ref_file = 'exp6_refe.txt'
        elif 'exp_07' in filename or 'exp7' in filename: ref_file = 'exp7_refe.txt'
        
        ref_times = None
        ref_values = None
        
        if ref_file:
             ref_path = os.path.join(project_root, ref_file)
             if os.path.exists(ref_path):
                 print(f"  Loading reference data from: {ref_file}")
                 try:
                     # Load Reference: Time, Cb, Gl, Xy (Assuming 4 cols)
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

        path_spec = os.path.join(project_root, filename)
        
        if not os.path.exists(path_spec):
            print(f"  File not found: {path_spec}")
            continue
            
        try:
            # Load Data with Header Parsing
            with open(path_spec, 'r') as f:
                header_line = f.readline().strip()
                
            header_parts = header_line.split()
            # Check if first part is 'Time' or similar
            if header_parts[0].lower() in ['time', 'index', 'timestamp']:
                # Wavelengths start from index 1 (Time col is index 0)
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                start_col_absor = 1
            else:
                # Assume all are wavelengths if no 'Time' label found (fallback)
                wl_curr = np.array([float(x) for x in header_parts])
                start_col_absor = 0
            
            # Load Data Block
            # Use pandas if available for speed? stick to numpy for deps consistency
            raw_data = np.loadtxt(path_spec, skiprows=1)
            
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
                
            if start_col_absor == 1:
                times = raw_data[:, 0]
                absor0 = raw_data[:, 1:]
            else:
                times = np.arange(raw_data.shape[0])
                absor0 = raw_data
            
            print(f"  Data Loaded: {absor0.shape[0]} samples, {absor0.shape[1]} wavelengths.")
            print(f"  Time range: {times[0]} to {times[-1]}")

            # Pretreatment
            print("  Applying pretreatment...")
            absor, wl_new = apply_pretreatment(pretreat, absor0, wl_curr, plot=False)
            
            # Prediction
            n_samples = absor.shape[0]
            y_pred = np.zeros((n_samples, 3))
            
            print("  Predicting...")
            for i, name in enumerate(cnames):
                y_p = models[name].predict(absor)
                # Single output model returns [N, 1] or [N, ]
                if y_p.ndim > 1:
                     y_pred[:, i] = y_p.flatten()
                else:
                     y_pred[:, i] = y_p
                
            # Save Predictions
            out_file_txt = os.path.join(output_dir, f"Pred_{filename}")
            header_out = "Time\tCb\tGl\tXy"
            data_out = np.column_stack([times, y_pred])
            np.savetxt(out_file_txt, data_out, header=header_out, delimiter='\t', fmt='%.4f')
            print(f"  Predictions saved to: {out_file_txt}")
            
            # Plot
            out_file_png = os.path.join(output_dir, f"Plot_{filename.replace('.txt', '.png')}")
            
            # Create subplots for each analyte
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            if not isinstance(axes, np.ndarray): axes = [axes]
            
            for i, name in enumerate(cnames):
                ax = axes[i]
                y_curr = y_pred[:, i]
                rmse = rmsecv_vals[i]
                
                ax.plot(times, y_curr, label=f'{name} (Pred)', color=colors[i], linewidth=2)
                
                # Fill between +/- RMSECV
                ax.fill_between(times, y_curr - rmse, y_curr + rmse, color=colors[i], alpha=0.2, label=f'±RMSECV ({rmse:.3f})')
                
                # Plot Reference Points if available
                if ref_times is not None and ref_values is not None:
                    ax.scatter(ref_times, ref_values[:, i], color=colors[i], edgecolors='black', marker='o', s=40, zorder=5, label='Offline Reference')
                
                ax.set_ylabel(f"{name} (g/L)")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
            axes[-1].set_xlabel("Time (min)")
            fig.suptitle(f"Predicted Concentrations - {filename}", fontsize=14)
            plt.tight_layout()
            
            plt.savefig(out_file_png, dpi=300)
            print(f"  Plot saved to: {out_file_png}")
            
            # Show interactive plot
            plt.show() # This will block until closed
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
