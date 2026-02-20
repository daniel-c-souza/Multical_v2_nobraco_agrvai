import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.analysis import func_analysis
plt.rcParams['figure.max_open_warning'] = 100
def main():
    # =========================================================================
    #                               CONFIGURATION
    # =========================================================================
    
    # --- 1. General Settings ---
    results_dir = "results"  # Directory to save results and plots
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. Model Parameters ---
    Selecao = 1       # Model type: 1 = PLS; 2 = SPA; 3 = PCR
    kmax = 15         # Maximum number of Latent Variables (Factors) for PLS/PCR
    
    # Analytes (Constituents)
    nc = 3            # Number of analytes
    cname = ['cb', 'gl', 'xy']  # Names of analytes (e.g. ['Glucose', 'Ethanol'])
    colors = ['green', 'red', 'purple'] # Colors for each analyte validation plot
    unid = 'g/L'      # Unit of measurement

    # Optimization Parameters (SPA/PCR specifics if needed)
    optkini = 2       # 0=> lini = lambda(1); 1=> lini = below; 2=> optimize lini.
    lini = 0          # [nm] Initial wavelength (only if optkini=1)

    # --- 3. Cross-Validation & Validation Settings ---
    # Test Set Split
    frac_test = 0.0   # Fraction of data to separate for pure testing (hold-out)
    dadosteste = []   # Manual test set (X_test, Y_test) if available, else []
    
    # Cross-Verification (CV) Settings
    # Mode: 'kfold' or 'Val' (Holdout)
    validation_mode = 'kfold' 
    
    # If 'kfold':
    kpart = 5                 # Number of folds for CV (if 'kfold')
    cv_type = 'venetian'      # Type of CV: 'random', 'consecutive', 'venetian'
    
    # If 'Val' (Holdout):
    frac_val = 0.20           # Fraction for validation holdout (0.0 - 1.0)
    
    # Construct OptimModel based on above settings
    if validation_mode == 'kfold':
        OptimModel = ['kfold', kpart, cv_type]
    elif validation_mode == 'Val':
        OptimModel = ['Val', frac_val]
    else:
         OptimModel = ['kfold', 5, 'venetian'] # Default


    # --- 4. Statistical Analysis & Outliers ---
    outlier = 0       # 0 = No Outlier Removal, 1 = Yes (Student t-test on residuals)
    use_ftest = True  # Use Osten F-test for automatic model selection (Optimal k)
    
    # Analyses to run before calibration
    # Options: 'LB' (Lambert-Beer), 'PCA' (Principal Component Analysis)
    analysis_list = [['LB'], ['PCA']]

    # --- 5. Data Files ---
    # List of tuples: (Concentration_File, Absorbance_File)
    # Concentration File: Text file with Y values (Time, Conc) or (Conc)
    # Absorbance File: Text file with Spectra (Time, WL1, WL2...)
    data_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),   
        ('exp7_refe.txt', 'exp7_nonda.txt'),
    ]
    # Note: Ensure these files exist in the working directory.

    # --- 6. Pretreatment Pipeline ---
    # List of [Operation, Parameters...]
    # Available Operations:
    #  ['Cut', min_wl, max_wl, plot_bool] : Cut spectral region
    #  ['SG', window, order, deriv, plot_bool] : Savitzky-Golay Smoothing/Deriv
    #  ['MA', window, order, plot_bool] : Moving Average
    #  ['Loess', alpha, order, plot_bool] : LOESS Smoothing
    #  ['Deriv', order, plot_bool] : Simple Derivative
    #  ['MSC', plot_bool] : Multiplicative Scatter Correction (if implemented)
    #  ['SNV', plot_bool] : Standard Normal Variate (if implemented)
    
    pretreat = [
        ['Cut', 4500, 8000, 1],
        
        ['SG', 7, 1, 2, 1, 1],
    ]

    # =========================================================================
    #                       DATA LOADING
    # =========================================================================

    x_list = []
    absor_list = []
    time_list_conc = [] # Store time from concentration file
    time_list_spec = [] # Store time from spectra file
    wavelengths = None
    
    print("Loading data...")
    for x_f, abs_f in data_files:
        if os.path.exists(x_f) and os.path.exists(abs_f):
            print(f"  - {x_f} / {abs_f}")
            try:
                xi = np.loadtxt(x_f)
                
                # Load absorbance file with header handling
                # First line: Time \t WL1 \t WL2 ...
                with open(abs_f, 'r') as f_node:
                    header_parts = f_node.readline().strip().split()
                
                # Extract wavelengths (skip "Time")
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                
                # Load data (skip header row)
                absi = np.loadtxt(abs_f, skiprows=1)
                
                # Check for Time column in Concentration File (assuming Col 0 is Time, Col 1 is Conc)
                # If 2 columns, treat Col 0 as Time.
                if xi.ndim == 2 and xi.shape[1] > 1:
                    ti_conc = xi[:, 0]
                    xi = xi[:, 1:] # Keep remaining columns as concentration
                else:
                    ti_conc = None
                    # Ensure xi is column vector (N, nc)
                    if xi.ndim == 1:
                        xi = xi.reshape(-1, 1)
                
                # Assuming new structure:
                # wl_curr loaded above from header
                data_curr = absi[:, 1:] # Skip first col (time) from data matrix
                ti_spec = absi[:, 0] # First col is time from data matrix
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
                    # Checking consistency
                    if not np.allclose(wavelengths, wl_curr, atol=1e-1):
                         print(f"Warning: Wavelengths in {abs_f} differ from previous.")
                
                x_list.append(xi)
                absor_list.append(data_curr)
                if ti_conc is not None:
                     time_list_conc.append(ti_conc)
                time_list_spec.append(ti_spec)
                
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")
        else:
            print(f"  [Skipping] Files not found: {x_f} or {abs_f}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        # Combine: Row 0 is wavelengths
        absor0 = np.vstack([wavelengths, absor_data])
        
        # Consolidate time if needed
        if time_list_conc:
             times_conc = np.concatenate(time_list_conc)
        
        print(f"Total samples loaded: {x0.shape[0]}")
    else:
        raise FileNotFoundError("No valid data files loaded. Please check data_files paths and ensure files exist.")

    
    # =========================================================================
    #                                EXECUTION
    # =========================================================================
    
    engine = MulticalEngine()
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
        frac_test, dadosteste, OptimModel, pretreat, 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier, use_ftest=use_ftest,
        colors=colors
    )
    
    model_map = {1: "PLS", 2: "SPA", 3: "PCR"}
    model_name = model_map.get(Selecao, "Unknown")
    
    if RMSECV is not None:
        print(f"\nResults for Model: {model_name}")
        print("RMSECVn (Normalized) (k=1..kmax):")
        print(RMSECV)
        print("\nRMSECVconc (Original Units) (k=1..kmax):")
        print(RMSECV_conc)
        print("\nRMSEcaln (Normalized) (k=1..kmax):")
        print(RMSEcal)
        print("\nRMSEcalconc (Original Units) (k=1..kmax):")
        print(RMSEcal_conc)
        
        if np.any(RMSEtest):
             print("\nRMSEtestn (Normalized) (k=1..kmax):")
             print(RMSEtest)
             print("\nRMSEtestconc (Original Units) (k=1..kmax):")
             print(RMSEtest_conc)
    
    print("\nProcessing complete. Showing plots...")
    plt.show()

if __name__ == "__main__":
    main()
