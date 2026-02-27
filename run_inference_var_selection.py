import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import load_and_predict_pls

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
RESULTS_DIR = "results_inference_var_sel" # Directory for output
MODEL_PATH = "results_var_selection/model_var_selection.pkl" # Path to trained model

# --- 2. Data Files ---
INFERENCE_FILES = [
        ('data/exp4_refe.txt', 'data/exp_04_inf.txt'),
        ('data/exp5_refe.txt', 'data/exp_05_inf.txt'),
        ('data/exp6_refe.txt', 'data/exp_06_inf.txt'),
        ('data/exp7_refe.txt', 'data/exp_07_inf.txt'),
]

# --- 3. Model Parameters ---
ANALYTES = ['cb', 'gl', 'xy']
COLORS = ['green', 'red', 'purple']
UNITS = 'g/L'
RMSECV_DUMMY = [0.0, 0.0, 0.0]

# --- 4. Pretreatment Pipeline ---
# Must match the variable selection model EXACTLY
PRETREATMENT = [
    ['Cut', 4400, 7500, 1],
    ['SG', 7, 2, 1, 1],
]

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def load_inference_data(files, nc):
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    time_list = []

    print("Loading inference data...")
    
    for x_f, abs_f in files:
        if not os.path.exists(abs_f): continue
        try:
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            start_idx = 1
            wl = np.array([float(x) for x in header[start_idx:]])
                
            absi_full = np.loadtxt(abs_f, skiprows=1)
            ti_spec = absi_full[:, 0]
            absi = absi_full[:, 1:]

            if wavelengths is None: wavelengths = wl
            
            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            time_list.append(ti_spec)
            
            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
                    
                    if ref_data.ndim > 1 and ref_data.shape[1] >= nc + 1:
                        # Match reference times
                        t_ref = ref_data[:, 0]
                        vals_ref = ref_data[:, 1:nc+1]
                        
                        for i, t_val in enumerate(t_ref):
                            t_val_min = t_val 
                            idx = (np.abs(ti_spec - t_val_min)).argmin()
                            if np.abs(ti_spec[idx] - t_val_min) < 5.0:
                                xi_aligned[idx, :] = vals_ref[i, :]
                                
                except Exception as e:
                    print(f"Warning loading ref {x_f}: {e}")
            
            x_list.append(xi_aligned)

        except Exception as e:
            print(f"Error loading {abs_f}: {e}")

    if not x_list: return None, None, None, []
    
    xinf0 = np.vstack(x_list)
    absorinf_data = np.vstack(absor_list)
    absorinf0 = np.vstack([wavelengths, absorinf_data])
    tinf0 = np.hstack(time_list)
    
    return xinf0, absorinf0, tinf0, sizes

def filter_to_model_wavelengths(absor_pre, wl_pre, wl_model):

    
    # Tolerant search for indices
    indices = []
    found_count = 0
    tolerance = 1e-1
    
    for wm in wl_model:
        diffs = np.abs(wl_pre - wm)
        idx = np.argmin(diffs)
        if diffs[idx] < tolerance:
            indices.append(idx)
            found_count += 1
        else:
            print(f"Warning: Wavelength {wm:.2f} expected by model not found in inference data!")
            # This is critical if we miss one
    
    if found_count != len(wl_model):
        raise ValueError("Could not align inference wavelengths to model selection.")
        
    absor_filtered = absor_pre[:, indices]
    wl_filtered = wl_pre[indices]
    
    return absor_filtered, wl_filtered

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams['figure.max_open_warning'] = 100

    # 1. Load Data
    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None: return
    
    wl_inf = absorinf0[0, :]
    absor_inf_raw = absorinf0[1:, :]

    # 2. Pretreatment
    print("Applying Pretreatment...")
    absor_inf_pre, wl_inf_pre = apply_pretreatment(PRETREATMENT, absor_inf_raw, wl_inf, plot=False)

    # 3. Load Model Metadata for Filtering
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please run run_variable_selection.py first.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    wl_model = model_data['wavelengths']
    
    # 4. Filter Wavelengths
    absor_inf_sel, wl_inf_sel = filter_to_model_wavelengths(absor_inf_pre, wl_inf_pre, wl_model)

    # 5. Predict
    try:
        X_pred, rmsecv_loaded = load_and_predict_pls(absor_inf_sel, wl_inf_sel, MODEL_PATH) 
        # Note: load_and_predict_pls will check again but it should pass now
    except Exception as e:
        print(f"Prediction Error: {e}")
        return
    
    if X_pred is not None:
        print(f"\nInference Complete. Predictions shape: {X_pred.shape}")
        if rmsecv_loaded:
             print(f"Loaded RMSECV from model: {rmsecv_loaded}")
        else:
             print("No RMSECV found in model (using default 0.0)")
             rmsecv_loaded = [0.0]*X_pred.shape[1]
        
        # Save X_pred
        output_matrix = np.column_stack([tinf0, X_pred])
        header = "Time\t" + "\t".join(ANALYTES)
        np.savetxt(os.path.join(RESULTS_DIR, "Predicted_Inference.txt"), output_matrix, delimiter='\t', header=header, comments='')

        # =========================================================================
        #                            TIME SERIES PLOTTING
        # =========================================================================
        print("\nCreating Time Series Plots...")
        
        current_idx = 0
        nc = X_pred.shape[1] 
        inf_files_plot = INFERENCE_FILES

        for i, size in enumerate(inf_sizes):
            end_idx = current_idx + size
            
            t_exp = tinf0[current_idx:end_idx] / 60.0
            pred_exp = X_pred[current_idx:end_idx, :]
            
            if xinf0 is not None:
                ref_exp = xinf0[current_idx:end_idx, :]
            else:
                ref_exp = None
                
            fname_ref, fname_spec = inf_files_plot[i] if i < len(inf_files_plot) else ("None", f"Exp_{i+1}")
            exp_name = os.path.basename(fname_spec)

            print(f"  Plotting Experiment: {exp_name} ({size} samples)")

            for j in range(nc):
                analyte_name = ANALYTES[j]
                
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                
                # Using loaded RMSECV
                rmse_entry = rmsecv_loaded[j] if j < len(rmsecv_loaded) else 0.0
                
                # Handle array if saved incorrectly as vector
                if np.ndim(rmse_entry) > 0 and len(rmse_entry) == nc:
                     rmse_val = rmse_entry[j]
                elif np.ndim(rmse_entry) > 0:
                     rmse_val = rmse_entry[0]
                else:
                     rmse_val = rmse_entry

                c_curr = COLORS[j] if j < len(COLORS) else 'blue'
                
                ax.fill_between(t_exp, pred_exp[:, j] - rmse_val, pred_exp[:, j] + rmse_val, 
                                color=c_curr, alpha=0.1, label=rf'RMSECV ($\pm${rmse_val:.2f})')
                
                ax.plot(t_exp, pred_exp[:, j], color=c_curr, linestyle='-', linewidth=2, label=f'Predicted {analyte_name}')
                
                if ref_exp is not None and ref_exp.shape[1] > j:
                    ref_vals = ref_exp[:, j]
                    mask_valid = ~np.isnan(ref_vals)
                    if np.any(mask_valid):
                        ax.scatter(t_exp[mask_valid], ref_vals[mask_valid], 
                                   color=c_curr, edgecolors='k', s=60, marker='o', zorder=5, label=f'Reference {analyte_name}')
                    
                ax.set_title(f"Inference: {analyte_name} | File: {exp_name}")
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(f"Concentration ({UNITS})")
                ax.legend(loc='best')
                ax.grid(False)
                
                safe_name = exp_name.replace('.txt', '').replace('.csv', '')
                plt.savefig(os.path.join(RESULTS_DIR, f"Plot_{safe_name}_{analyte_name}.png"))
                plt.close(fig) 
            
            current_idx = end_idx
            
        # =========================================================================
        #                            PARITY PLOTS (Pred vs Ref)
        # =========================================================================
        print("\nCreating Parity Plots (Pred vs Ref)...")
        if xinf0 is not None:
             # Calculate Metrics and Plot for ALL inference data combined
             from sklearn.metrics import r2_score, mean_squared_error

             for j in range(nc):
                analyte_name = ANALYTES[j]
                
                # Get all pred and ref for this analyte
                y_pred_all = X_pred[:, j]
                y_ref_all = xinf0[:, j]
                
                # Filter NaNs in Reference
                mask = ~np.isnan(y_ref_all)
                if np.sum(mask) == 0:
                    print(f"  Skipping Parity Plot for {analyte_name}: No reference data.")
                    continue
                    
                y_pred_valid = y_pred_all[mask]
                y_ref_valid = y_ref_all[mask]
                
                # Metrics
                rmsep = np.sqrt(mean_squared_error(y_ref_valid, y_pred_valid))
                r2 = r2_score(y_ref_valid, y_pred_valid)
                
                # Plot
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(y_ref_valid, y_pred_valid, color=COLORS[j], edgecolors='k', alpha=0.7)
                
                # 1:1 Line
                min_val = min(y_ref_valid.min(), y_pred_valid.min())
                max_val = max(y_ref_valid.max(), y_pred_valid.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
                
                plt.title(f"{analyte_name}: Pred vs Ref (RMSEP={rmsep:.2f}, R2={r2:.2f})")
                plt.xlabel(f"Reference ({UNITS})")
                plt.ylabel(f"Predicted ({UNITS})")

                plt.grid(True)
                
                plt.savefig(os.path.join(RESULTS_DIR, f"Parity_Plot_{analyte_name}.png"))
                plt.close(fig)
                print(f"  Saved Parity Plot for {analyte_name} (RMSEP={rmsep:.4f})")
                
    print("\nProcessing complete. Plots saved.")

if __name__ == "__main__":
    main()
