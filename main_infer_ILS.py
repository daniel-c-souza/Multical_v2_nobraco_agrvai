import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.inference import run_inference

def load_data_from_list(file_pairs, nc=3):
    """
    Helper to load pairs of (Concentration, Absorbance) files.
    """
    x_list = []
    absor_list = []
    time_list = []
    wavelengths = None
    
    for x_f, abs_f in file_pairs:
        if os.path.exists(x_f) and os.path.exists(abs_f):
            print(f"  - Loading {x_f} / {abs_f}")
            try:
                # Load Concentration (reference)
                xi = np.loadtxt(x_f)
                
                # Load Absorbance (spectra) - assuming header row exists
                try:
                    # Try loading with header skip first
                    absi_full = np.loadtxt(abs_f, skiprows=1)
                    # We need to read the header to get wavelengths
                    with open(abs_f, 'r') as f:
                        header_line = f.readline().strip().split()
                        # header_line[0] is "Time", header_line[1:] are wavelengths
                        wl_curr = np.array([float(w) for w in header_line[1:]])
                        
                    ti_spec = absi_full[:, 0]
                    # Ensure ti_spec is 1D
                    if ti_spec.ndim > 1: ti_spec = ti_spec.flatten()
                    
                    data_curr = absi_full[:, 1:]
                    
                except ValueError:
                    # Fallback if no text header or different format
                    absi = np.loadtxt(abs_f)
                    if absi.shape[0] == 0: continue
                    wl_curr = absi[0, :]
                    data_curr = absi[1:, :]
                    ti_spec = None

                # Check for Time column in Concentration File
                ti_conc = None
                
                # Handle 1D array for xi -> (samples, 1)
                if xi.ndim == 1:
                     xi = xi.reshape(-1, 1)
                
                # exp4_refe.txt: Col 0 is Index/Time? 
                # If we assume Col 0 is index, valid data is xi[:, 1:]
                # Let's assume Col 0 is NOT concentration.
                if xi.shape[1] > nc: 
                    xi = xi[:, 1:]

                n_conc = xi.shape[0]
                n_spec = data_curr.shape[0]
                
                if n_conc != n_spec:
                    print(f"Warning: Sample count mismatch in {x_f} ({n_conc}) vs {abs_f} ({n_spec}). Trimming to min.")
                    min_len = min(n_conc, n_spec)
                    xi = xi[:min_len, :]
                    data_curr = data_curr[:min_len, :]
                    if ti_spec is not None: ti_spec = ti_spec[:min_len]
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
                    if len(wavelengths) == len(wl_curr) and not np.allclose(wavelengths, wl_curr, atol=1e-1):
                         print(f"Warning: Wavelengths in {abs_f} differ from previous.")
                
                x_list.append(xi)
                absor_list.append(data_curr)
                if ti_spec is not None:
                     time_list.append(ti_spec)
                     
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  [Skipping] Files not found: {x_f} or {abs_f}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        absor0 = np.vstack([wavelengths, absor_data])
        
        t0 = None
        if len(time_list) > 0:
            t0 = np.hstack(time_list) # Time is usually 1D
            
        return x0, absor0, t0
    else:
        return None, None, None

def load_inference_data(file_pairs, nc=3):
    """
    Loads inference files avoiding truncation of spectral data.
    Aligns reference data to spectral time points if possible, otherwise fills with NaNs.
    """
    x_list = []
    absor_list = []
    time_list = []
    sizes = []
    wavelengths = None
    
    for x_f, abs_f in file_pairs:
        if os.path.exists(abs_f):
            print(f"  - Loading Inference Spectra {abs_f}")
            try:
                # 1. Load Spectra (The "Master" Timeline)
                try:
                    # Attempt to read header for wavelengths
                    with open(abs_f, 'r') as f:
                        header_line = f.readline().strip().split()
                        # Check if header looks like "Time 1200 1201..."
                        if header_line[0] in ['Time', 'time', 'Wavenumber', 'wavenumber']:
                             wl_curr = np.array([float(w) for w in header_line[1:]])
                        else:
                        #  No standard header? Fallback
                             raise ValueError("No standard header found")
                    
                    absi_full = np.loadtxt(abs_f, skiprows=1)
                    ti_spec = absi_full[:, 0]
                    data_curr = absi_full[:, 1:]
                    
                except Exception:
                    # Fallback
                    absi = np.loadtxt(abs_f)
                    if absi.shape[0] == 0: continue
                    wl_curr = absi[0, :]
                    data_curr = absi[1:, :]
                    ti_spec = np.arange(data_curr.shape[0]) # Dummy time

                if wavelengths is None:
                    wavelengths = wl_curr
                
                n_samples = data_curr.shape[0]
                sizes.append(n_samples)
                
                # 2. Load Reference (If exists) and Align
                xi_aligned = np.full((n_samples, nc), np.nan) 
                
                if os.path.exists(x_f):
                    print(f"    (Found reference file {x_f} - Aligning...)")
                    try:
                        ref_data = np.loadtxt(x_f)
                        # Assume ref_data structure: [Time, Ref1, Ref2, Ref3]
                        
                        if ref_data.ndim == 1: ref_data = ref_data.reshape(1, -1)
                        
                        # Check dimensions
                        if ref_data.shape[1] >= nc + 1:
                            t_ref = ref_data[:, 0]
                            vals_ref = ref_data[:, 1:nc+1]
                            
                            # Matching: Find nearest time point in spectra for each ref point
                            for i, t_val in enumerate(t_ref):
                                # Find index in ti_spec closest to t_val
                                # Convert Ref (Hours) to match Spec (Minutes)
                                t_val_min = t_val * 60.0 
                                idx = (np.abs(ti_spec - t_val_min)).argmin()
                                xi_aligned[idx, :] = vals_ref[i, :]
                        else:
                            print(f"    Warning: Reference file {x_f} has {ref_data.shape[1]} columns. Expected > {nc}. usage: [Time, Ref1, Ref2...]. Skipping alignment.")
                    except Exception as e:
                        print(f"    Error reading reference {x_f}: {e}")

                absor_list.append(data_curr)
                time_list.append(ti_spec)
                x_list.append(xi_aligned)
                    
            except Exception as e:
                print(f"Error loading {abs_f}: {e}")
                import traceback
                traceback.print_exc()

    if x_list:
        absor_data = np.vstack(absor_list)
        absor0 = np.vstack([wavelengths, absor_data])
        
        t0 = np.hstack(time_list)
        xinf0 = np.vstack(x_list)
            
        return xinf0, absor0, t0, sizes
    else:
        return None, None, None, []

def main():
    # =========================================================================
    #                                   CONFIG
    # =========================================================================
    
    results_dir = "results_inference"
    os.makedirs(results_dir, exist_ok=True)
    
    # Method
    Selecao = 1 # 1 = PLS; 2 = SPA; 3=PCR
    optkini = 2 # SPA Opt: 0=> lini = lambda(1); 1=> lini = given; 2=> optimize
    lini = 0 

    # Model Complexity for Inference
    # Number of LVs for each analyte. Must match 'nc'.
    # Example: [k_analyte1, k_analyte2, k_analyte3]
    kinf = [4,5,5] 
    
    # Analytes
    nc = 3
    cname = ['cb', 'gl', 'xy'] 
    colors =  ['green', 'red', 'purple'] 
    unid = 'g/L'
    
    # =========================================================================
    #                                   DATA
    # =========================================================================
    
    # --- Calibration Data ---
    # Files to build the model
    cal_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),   
        #('x03.txt', 'abs03.txt'),
        # ('x04.txt', 'abs04.txt') 
    ]
    
    # --- Inference Data ---
    # Files to predict on
    inf_files = [
        ('exp4_refe.txt', 'exp_04_inf.txt'),
        ('exp5_refe.txt', 'exp_05_inf.txt'),
        ('exp6_refe.txt', 'exp_06_inf.txt'),  
        # ('x05.txt', 'abs05.txt')
    ]
    
    print("Loading Calibration Data...")
    x0, absor0, t0 = load_data_from_list(cal_files)
    if x0 is None:
        print("Error: No calibration data loaded.")
        return

    print("Loading Inference Data...")
    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(inf_files)
    if absorinf0 is None:
        print("Error: No inference data loaded.")
        return

    # =========================================================================
    #                                PRETREATMENT
    # =========================================================================
    
    # Pretreatment for Calibration
    pretreat = [
        ['Cut', 4500, 8000 , 1],
        ['SG',7,1,2,1,1]
    ]
    
    # Pretreatment for Inference (usually same as Cal)
    # But can be different if needed (e.g. slight baseline shift handling)
    pretreatinf = list(pretreat) 
    # Example adaptation:
    # pretreatinf = [['Cut', 4500, 10000, 1], ['SG', 3, 1, 5, 1, 1]]

    # =========================================================================
    #                                ANALYSIS
    # =========================================================================
    
    analysis_list = [['LB'], ['PCA']]
    analysisinf_list = [] # [['PCA']]

    # Leverage Analysis
    leverage = 0 

    # =========================================================================
    #                                EXECUTION
    # =========================================================================
    
    X_pred, rmsecv = run_inference(
        Selecao, optkini, lini, kinf, nc, cname, unid, 
        x0, absor0, xinf0, absorinf0, 
        pretreat, pretreatinf, 
        analysis_list=analysis_list, analysisinf_list=analysisinf_list, 
        leverage=leverage, output_dir=results_dir, colors=colors
    )
    
    # =========================================================================
    #                            TIME SERIES PLOTTING
    # =========================================================================
    
    if X_pred is not None:
        print("\nCreating Time Series Plots...")
        
        # Determine Time Axis Global
        if tinf0 is None or len(tinf0) != X_pred.shape[0]:
            print("Warning: No valid time vector found for inference. Using Sample Index.")
            tinf0 = np.arange(X_pred.shape[0])
            xlabel_text_base = "Sample Index"
        else:
            xlabel_text_base = "Time"

        # Split predictions by experiment
        current_idx = 0
        
        # Ensure we don't go out of bounds if sizes mismatch (sanity check)
        if sum(inf_sizes) != X_pred.shape[0]:
             print(f"Warning: Total samples {X_pred.shape[0]} does not match sum of file sizes {sum(inf_sizes)}. Plotting concatenated.")
             inf_sizes = [X_pred.shape[0]]
             inf_files_plot = [("All", "All")]
        else:
             inf_files_plot = inf_files

        for i, size in enumerate(inf_sizes):
            end_idx = current_idx + size
            
            # Extract data for this experiment
            t_exp = tinf0[current_idx:end_idx] / 60.0 # Convert minutes to hours for plotting
            pred_exp = X_pred[current_idx:end_idx, :]
            
            if xinf0 is not None:
                ref_exp = xinf0[current_idx:end_idx, :]
            else:
                ref_exp = None
                
            # File name for title
            if i < len(inf_files):
                fname_ref, fname_spec = inf_files[i]
                exp_name = os.path.basename(fname_spec)
            else:
                exp_name = f"Exp_{i+1}"

            print(f"  Plotting Experiment: {exp_name} ({size} samples)")

            for j in range(nc):
                analyte_name = cname[j]
                
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)
                
                # Prediction Error Band (Global RMSECV)
                rmse_val = rmsecv[j]
                c_curr = colors[j] if j < len(colors) else 'blue'
                
                ax.fill_between(t_exp, pred_exp[:, j] - rmse_val, pred_exp[:, j] + rmse_val, 
                                color=c_curr, alpha=0.1, label=rf'RMSECV ($\pm${rmse_val:.2f})')
                
                # Predicted (Line)
                ax.plot(t_exp, pred_exp[:, j], color=c_curr, linestyle='-', linewidth=2, label=f'Predicted {analyte_name}')
                
                # Reference (Scatter) - Only plot non-NaNs
                if ref_exp is not None and ref_exp.shape[1] > j:
                    ref_vals = ref_exp[:, j]
                    # Filter NaNs
                    mask_valid = ~np.isnan(ref_vals)
                    if np.any(mask_valid):
                        ax.scatter(t_exp[mask_valid], ref_vals[mask_valid], 
                                   color=c_curr, edgecolors='k', s=60, marker='o', zorder=5, label=f'Reference {analyte_name}')
                    
                ax.set_title(f"Inference: {analyte_name} | File: {exp_name}")
                ax.set_xlabel("Time (h)")
                ax.set_ylabel(f"Concentration ({unid})")
                ax.legend(loc='best')
                ax.grid(False)
                
                # Save
                safe_name = exp_name.replace('.txt', '').replace('.csv', '')
                plt.savefig(os.path.join(results_dir, f"Plot_{safe_name}_{analyte_name}.png"))
                # plt.close(fig) # Close to avoid memory buildup
            
            current_idx = end_idx
            
    print("\nProcessing complete. plots saved to results folder.")
    plt.show() 

if __name__ == "__main__":
    main()
