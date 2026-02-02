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
    print("=== ANN Inference ===")
    
    # 1. Configuration
    models_dir = os.path.join(project_root, "models", "ann_best")
    output_dir = os.path.join(project_root, "results_inference_ann")
    os.makedirs(output_dir, exist_ok=True)
    
    cnames = ['cb', 'gl', 'xy']
    
    # Must match training exactly!
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 1, 2, 1, 1],
    ]
    
    # Files to predict
    # Format: (Spectral File, Reference File or None, Name)
    files_to_predict = [
        ('exp4_nonda.txt', 'exp4_refe.txt', 'Exp 04'),
        ('exp5_nonda.txt', 'exp5_refe.txt', 'Exp 05'),
        ('exp6_nonda.txt', 'exp6_refe.txt', 'Exp 06'),
        ('exp7_nonda.txt', 'exp7_refe.txt', 'Exp 07')
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
        except Exception as e:
            print(f"Failed to load model for {name}: {e}")
            return
            
    # 3. Process Each File
    for spec_file, ref_file, label in files_to_predict:
        print(f"\nProcessing {label} ({spec_file})...")
        path_spec = os.path.join(project_root, spec_file)
        
        if not os.path.exists(path_spec):
            print(f"  File not found: {path_spec}")
            continue
            
        try:
            # Load Spectra
            with open(path_spec, 'r') as f_node:
                header_parts = f_node.readline().strip().split()
            
            # Check if header is numerical or text
            try:
                # Try parsing wavelengths from header
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                start_skip = 1
            except ValueError:
                # Header might be text, try second line?
                # Usually our format has wavelengths in first line.
                # If parsing fails, assuming standard format or need check
                print("  Warning: Header parsing issue. Assuming standard structure.")
                start_skip = 1
                
            raw_data = np.loadtxt(path_spec, skiprows=start_skip)
            
            # Handle timestamps/indices in first column
            if raw_data.ndim == 2:
                absor0 = raw_data[:, 1:]
            else:
                absor0 = raw_data.reshape(1, -1)
                
            # Pretreatment
            absor, wl_new = apply_pretreatment(pretreat, absor0, wl_curr, plot=False)
            
            # Prediction
            n_samples = absor.shape[0]
            y_pred = np.zeros((n_samples, 3))
            
            for i, name in enumerate(cnames):
                y_p = models[name].predict(absor)
                # If model is multi-output (trained on 3 targets), select the specific target 'i'
                if y_p.ndim > 1 and y_p.shape[1] > 1:
                     y_pred[:, i] = y_p[:, i]
                else:
                     y_pred[:, i] = y_p.flatten()
                
            # Save Predictions
            out_file = os.path.join(output_dir, f"Pred_{spec_file}")
            header_out = "Cb\tGl\tXy"
            np.savetxt(out_file, y_pred, header=header_out, fmt='%.4f', delimiter='\t')
            print(f"  Predictions saved to {out_file}")
            
            # Compare with Ref if available
            path_ref = os.path.join(project_root, ref_file) if ref_file else None
            
            if path_ref and os.path.exists(path_ref):
                print(f"  Found reference file: {ref_file}")
                # Assuming Ref file aligns with Spec file
                # Users Ref files often have a first column of indices too
                y_ref_raw = np.loadtxt(path_ref)
                if y_ref_raw.ndim == 2:
                    y_ref = y_ref_raw[:, 1:] 
                else: 
                     y_ref = y_ref_raw # Or handle as error
                
                # Truncate to min length if mismatch
                min_len = min(len(y_pred), len(y_ref))
                y_pred_c = y_pred[:min_len]
                y_ref_c = y_ref[:min_len]
                
                # RMSE
                rmse = np.sqrt(np.mean((y_pred_c - y_ref_c)**2, axis=0))
                print(f"  RMSE Prediction: Cb={rmse[0]:.4f}, Gl={rmse[1]:.4f}, Xy={rmse[2]:.4f}")
                
                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for j, name in enumerate(cnames):
                    ax = axes[j]
                    ax.plot(y_ref_c[:, j], label='Measured', color='black')
                    ax.plot(y_pred_c[:, j], label='Predicted', color='red', linestyle='--')
                    ax.set_title(f"{label} - {name} (RMSE={rmse[j]:.4f})")
                    ax.legend()
                    
                plot_file = os.path.join(output_dir, f"Plot_{label}.png")
                fig.savefig(plot_file)
                # plt.show()
                plt.close(fig)
                
        except Exception as e:
            print(f"  Error processing {spec_file}: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. Global Calibration Plot
    # Gather all available predictions from the lists we just processed?
    # Actually, proper way is to keep a list
    global_y_pred = []
    global_y_ref = []
    
    # We need to re-loop or store them. Let's make the loop above append to these lists.
    # Refactoring slightly to just do it in a fresh loop or during the collection.
    
    print("\n--- Generating Global Calibration Plot (Exp 4-7) ---")
    
    all_y_ref = []
    all_y_pred = []
    
    for spec_file, ref_file, label in files_to_predict:
         path_spec = os.path.join(project_root, spec_file)
         path_ref = os.path.join(project_root, ref_file)
         
         if os.path.exists(path_spec) and os.path.exists(path_ref):
             try:
                 # Load Spec
                 with open(path_spec, 'r') as f_node:
                     header_parts = f_node.readline().strip().split()
                 wl_curr = np.array([float(x) for x in header_parts[1:]])
                 raw_data = np.loadtxt(path_spec, skiprows=1)
                 if raw_data.ndim == 2:
                    absor0 = raw_data[:, 1:]
                 else:
                    absor0 = raw_data.reshape(1, -1)
                 
                 absor, _ = apply_pretreatment(pretreat, absor0, wl_curr, plot=False)
                 
                 # Load Ref
                 y_ref_i = np.loadtxt(path_ref)
                 if y_ref_i.ndim == 2: y_ref_i = y_ref_i[:, 1:]
                 
                 # Predict
                 n_samples = absor.shape[0]
                 y_pred_i = np.zeros((n_samples, 3))
                 for i, name in enumerate(cnames):
                    y_p = models[name].predict(absor)
                    y_pred_i[:, i] = y_p.flatten()
                    
                 all_y_ref.append(y_ref_i)
                 all_y_pred.append(y_pred_i)
             except Exception as e:
                 print(f"Skip {label} in global plot: {e}")
                 
    if all_y_ref:
        Y_ref_total = np.vstack(all_y_ref)
        Y_pred_total = np.vstack(all_y_pred)
        
        # Calculate Global RMSE
        rmse_global = np.sqrt(np.mean((Y_pred_total - Y_ref_total)**2, axis=0))
        print(f"Global RMSEC: Cb={rmse_global[0]:.4f}, Gl={rmse_global[1]:.4f}, Xy={rmse_global[2]:.4f}")
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for j, name in enumerate(cnames):
            ax = axes[j]
            
            # Scatter Plot
            ax.scatter(Y_ref_total[:, j], Y_pred_total[:, j], alpha=0.7, color='blue', edgecolors='k')
            
            # Diagonal Line
            min_val = min(Y_ref_total[:, j].min(), Y_pred_total[:, j].min())
            max_val = max(Y_ref_total[:, j].max(), Y_pred_total[:, j].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
            
            ax.set_title(f"{name} Global Cal (RMSEC={rmse_global[j]:.4f})")
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")
            ax.grid(True, linestyle=':', alpha=0.6)
            
        plot_file_global = os.path.join(output_dir, "Global_Calibration_Plot.png")
        fig.savefig(plot_file_global, dpi=300)
        print(f"Global calibration plot saved to {plot_file_global}")
        # plt.show()
        plt.close(fig)

if __name__ == "__main__":
    main()
