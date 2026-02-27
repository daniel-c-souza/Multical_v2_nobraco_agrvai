"""
=============================================================================
                            RUN INFERENCE SCRIPT
=============================================================================
This script performs inference (prediction) using a pre-trained PLS model.
It loads a saved model (.pkl), applies pretreatment, predicts analyte 
concentrations, and generates time-series plots comparing predictions to 
reference values (if available).

Workflow:
1. Load Configuration: Define paths, analytes, and pretreatment settings.
2. Load Inference Data: Read absorbance spectra and optional reference data.
3. Pretreatment: Apply the SAME pretreatment used during calibration.
4. Model Loading: Load the trained PLS model (coefficients, normalization).
5. Prediction: Predict Y values for the new spectra.
6. Plotting: Generate time-series plots with RMSECV confidence intervals.

Author: GitHub Copilot
Date: 2026-02-27
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.core.saving import load_and_predict_pls

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
# Directory where prediction results and plots will be saved
RESULTS_DIR = "results_inference" 

# Path to the trained calibration model file (.pkl)
MODEL_PATH = "results/model_calibration.pkl" 

# --- 2. Data Files ---
# List of (Reference_File, Absorbance_Spectra_File) pairs.
# Reference files are optional but required for validation plots.
# If no reference, use None or a dummy path (logic handles missing files).
# Ensure formats match: 
#   - Spectra: Rows=Samples, Cols=Wavelengths (1st row=Wavelengths or header)
#   - Reference: Rows=Samples, Cols=[Time, Ref1, Ref2...]
INFERENCE_FILES = [
    ('data/exp4_refe.txt', 'data/exp_04_inf.txt'),
    ('data/exp5_refe.txt', 'data/exp_05_inf.txt'),
    ('data/exp6_refe.txt', 'data/exp_06_inf.txt'),
    ('data/exp7_refe.txt', 'data/exp_07_inf.txt'),
]

# --- 3. Model Parameters ---
# Must match the calibration configuration
ANALYTES = ['cb', 'gl', 'xy']       # Names of analytes to predict
COLORS = ['green', 'red', 'purple'] # Colors for plotting each analyte
UNITS = 'g/L'                       # Concentration units

# --- 4. Pretreatment Pipeline ---
# CRITICAL: This must match the calibration model's pretreatment exactly.
# Format: [Method, Parameter1, Parameter2, ..., PlotFlag]
# Example: ['SG', Window=7, Poly=2, Ord=1, Plot=0]
PRETREATMENT = [
    ['Cut', 4400, 7500, 0],
    ['SG', 7, 2, 1, 0],
]

# =============================================================================
#                            HELPER FUNCTIONS
# =============================================================================

def load_inference_data(files, nc):
    """
    Loads absorbance spectra and aligns optional reference data.

    Args:
        files: List of (ref_path, spec_path) tuples.
        nc: Number of analytes (columns to read from reference file).

    Returns:
        xinf0: Aligned reference data (or None).
        absorinf0: Spectral data with wavelengths row.
        tinf0: Time vector (concatenated).
        sizes: List of sample counts per file (for splitting plots later).
    """
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    time_list = []

    print("Loading inference data...")
    
    for x_f, abs_f in files:
        if not os.path.exists(abs_f): continue
        try:
            # Load Spectra
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            
            # Try to handle "Time" or "Wavenumber" headers
            start_idx = 1
            wl = np.array([float(x) for x in header[start_idx:]])
                
            absi_full = np.loadtxt(abs_f, skiprows=1)
            
            # Assuming col 0 is Time/Index and remaining are spectral points
            ti_spec = absi_full[:, 0]
            absi = absi_full[:, 1:]

            if wavelengths is None: wavelengths = wl
            
            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            time_list.append(ti_spec)
            
            # Load Reference (if available) and align
            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    
                    if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
                    
                    # Check basic shape: [Time, Ref1, Ref2, Ref3...]
                    if ref_data.shape[1] >= nc + 1:
                        # Match reference times to spectral times
                        # Assumption: Reference Time is in MINUTES, Spectral Time is in MINUTES
                        t_ref = ref_data[:, 0]
                        vals_ref = ref_data[:, 1:nc+1]
                        
                        for i, t_val in enumerate(t_ref):
                            t_val_min = t_val 
                            
                            # Find index in ti_spec closest to t_val_min
                            idx = (np.abs(ti_spec - t_val_min)).argmin()
                            
                            # Optional: Check if time difference is acceptable (e.g. < 5 mins)
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

def main():
    """
    Main Execution Function.
    1. Loads data.
    2. Applies pretreatment.
    3. Loads model and predicts.
    4. Saves results.
    5. Plots time-series comparison.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams['figure.max_open_warning'] = 100

    # 1. Load Data
    # ----------------------------------------------------
    # Reads all absorbance files, concatenates them, and attempts to align
    # reference data based on timestamps (assuming minutes).
    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None: 
        print("No valid inference data found. Exiting.")
        return
    
    # Separate Wavelengths (Row 0) and Spectra (Rows 1+)
    wl_inf = absorinf0[0, :]
    absor_inf_raw = absorinf0[1:, :]

    # 2. Pretreatment
    # ----------------------------------------------------
    # Applies Savitzky-Golay, Cutting, etc. to match calibration.
    print("Applying Pretreatment...")
    absor_inf_pre, wl_inf_pre = apply_pretreatment(PRETREATMENT, absor_inf_raw, wl_inf, plot=False)

    # 3. Prediction
    # ----------------------------------------------------
    # Loads the saved PLS model (.pkl) and predicts Y values.
    # The loading function handles feature matching and normalization.
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please run run_calibration.py first.")
        return

    try:
        # Returns: (Predictions, List of RMSECV errors per analyte)
        X_pred, rmsecv_loaded = load_and_predict_pls(absor_inf_pre, wl_inf_pre, MODEL_PATH)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return
    
    if X_pred is not None:
        print(f"\nInference Complete. Predictions shape: {X_pred.shape}")
        
        # Determine RMSECV (Error Bars)
        # If available in the model, use it. Otherwise default to 0.0.
        if rmsecv_loaded:
             print(f"Loaded RMSECV from model: {rmsecv_loaded}")
        else:
             print("No RMSECV found in model (using default 0.0)")
             rmsecv_loaded = [0.0]*X_pred.shape[1]
             
        # Save Predictions
        print(f"Results saved to: {RESULTS_DIR}")
        # Format: [Time, Val1, Val2...]
        output_matrix = np.column_stack([tinf0, X_pred])
        header = "Time\t" + "\t".join(ANALYTES)
        np.savetxt(os.path.join(RESULTS_DIR, "Predicted_Inference.txt"), output_matrix, delimiter='\t', header=header, comments='')

        # =========================================================================
        #                            TIME SERIES PLOTTING
        # =========================================================================
        # Generate plots for each experiment (file) individually.
        print("\nCreating Time Series Plots...")

        current_idx = 0         # Tracks start row for current file
        nc = X_pred.shape[1]    # Number of analytes (columns)
        
        inf_files_plot = INFERENCE_FILES

        # Loop through each loaded file based on known sizes
        for i, size in enumerate(inf_sizes):
            end_idx = current_idx + size
            
            # Slice data for this experiment
            t_exp = tinf0[current_idx:end_idx] / 60.0 # Convert minutes to hours
            pred_exp = X_pred[current_idx:end_idx, :]
            
            # Slice reference data if available
            if xinf0 is not None:
                ref_exp = xinf0[current_idx:end_idx, :]
            else:
                ref_exp = None
                
            # Determine plot title from filename
            if i < len(inf_files_plot):
                fname_ref, fname_spec = inf_files_plot[i]
                exp_name = os.path.basename(fname_spec)
            else:
                exp_name = f"Exp_{i+1}"

            print(f"  Plotting Experiment: {exp_name} ({size} samples)")

            # Create plot for each analyte
            for j in range(nc):
                analyte_name = ANALYTES[j]
                
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                
                # --- RMSECV Handling ---
                # Retrieve the stored RMSECV for this analyte.
                # If stored as a single value (scalar) or list, extract safely.
                rmse_entry = rmsecv_loaded[j] if j < len(rmsecv_loaded) else 0.0
                
                # If the entry is an array (e.g. from saving conc vector), fallback to first element or scalar.
                if np.ndim(rmse_entry) > 0:
                    if len(rmse_entry) == nc:
                         # Likely means we saved [rmse_all_analytes] instead of scalar
                         rmse_val = rmse_entry[j]
                    else:
                         rmse_val = rmse_entry[0] # Fallback
                else:
                    rmse_val = rmse_entry

                # Set Color
                c_curr = COLORS[j] if j < len(COLORS) else 'blue'
                
                # Plot Confidence Interval (Pred +/- RMSECV)
                ax.fill_between(t_exp, pred_exp[:, j] - rmse_val, pred_exp[:, j] + rmse_val, 
                                color=c_curr, alpha=0.1, label=rf'RMSECV ($\pm${rmse_val:.2f})')
                
                # Plot Prediction Line
                ax.plot(t_exp, pred_exp[:, j], color=c_curr, linestyle='-', linewidth=2, label=f'Predicted {analyte_name}')
                
                # Plot Reference Points (if available)
                if ref_exp is not None and ref_exp.shape[1] > j:
                    ref_vals = ref_exp[:, j]
                    # Filter out NaN values in reference
                    mask_valid = ~np.isnan(ref_vals)
                    if np.any(mask_valid):
                        ax.scatter(t_exp[mask_valid], ref_vals[mask_valid], 
                                   color=c_curr, edgecolors='k', s=60, marker='o', zorder=5, label=f'Reference {analyte_name}')
                    
                # Formatting
                ax.set_title(f"Inference: {analyte_name} | File: {exp_name}")
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(f"Concentration ({UNITS})")
                ax.legend(loc='best')
                ax.grid(False)
                
                # Save Figure
                safe_name = exp_name.replace('.txt', '').replace('.csv', '')
                plt.savefig(os.path.join(RESULTS_DIR, f"Plot_{safe_name}_{analyte_name}.png"))
                plt.close(fig) 
            
            current_idx = end_idx
            
    print("\nProcessing complete. plots saved to results folder.")

if __name__ == "__main__":
    main()
