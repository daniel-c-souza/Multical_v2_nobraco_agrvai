import numpy as np
import os
import sys

# Add project root to sys.path to allow imports from src and svr_experiment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
from svr_experiment.svr_engine import MulticalEngineSVR  # Updated Class Name
from src.multical.analysis import func_analysis

plt.rcParams['figure.max_open_warning'] = 100

def main():
    # =========================================================================
    #                               CONFIGURATION
    # =========================================================================
    
    # --- 1. General Settings ---
    results_dir = "results_svr_experiment"  # Directory to save results and plots
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. Model Parameters ---
    Selecao = 4       # SVR
    kmax = 1000       # Grid Search Iterations (Covering ~760 combinations)
    
    # Analytes (Constituents)
    nc = 3            # Number of analytes
    cname = ['cb', 'gl', 'xy']  # Names
    colors = ['green', 'red', 'purple']       # Plot colors
    unid = 'g/L'      # Unit

    # Optimization Parameters (Unused for SVR but kept for signature compatibility)
    optkini = 0       
    lini = 0          

    # --- 3. Cross-Validation Settings ---
    # Test Set Split (Holdout)
    frac_test = 0.0   
    dadosteste = []   
    
    # Validation Mode
    validation_mode = 'kfold' 
    kpart = 5                 
    cv_type = 'random'      # SVR usually benefits from random shuffle CV

    # Construct OptimModel
    if validation_mode == 'kfold':
        OptimModel = ['kfold', kpart, cv_type]
    elif validation_mode == 'Val':
        # frac_val = 0.20
        # OptimModel = ['Val', frac_val]
        pass
    else:
         OptimModel = ['kfold', 5, 'venetian'] 

    # --- 4. Statistical Analysis ---
    outlier = 0       # 0 = No Outlier Removal
    use_ftest = False # Not used for SVR grid search
    
    analysis_list = [['LB'], ['PCA']]

    # --- 5. Data Files ---
    # Matches the user workspace structure
    data_files = [ 
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),
        ('exp7_refe.txt', 'exp7_nonda.txt'),
    ]

    # --- 6. Pretreatment Pipeline ---
    # Consistent with ANN experiment (Standard NIR Preprocessing)
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 2, 1, 1],
    ]

    # =========================================================================
    #                       DATA LOADING
    # =========================================================================

    x_list = []
    absor_list = []
    wavelengths = None
    
    print("Loading data...")
    for x_f, abs_f in data_files:
        if os.path.exists(x_f) and os.path.exists(abs_f):
            print(f"  - {x_f} / {abs_f}")
            try:
                # Load Concentration
                xi = np.loadtxt(x_f)
                
                # Load Spectra Header
                with open(abs_f, 'r') as f_node:
                    header_str = f_node.readline().strip()
                
                header_parts = header_str.split()
                # If first item is string "Time", skip it
                if header_parts[0].lower() == 'time':
                    wl_curr = np.array([float(x) for x in header_parts[1:]])
                else:
                    wl_curr = np.array([float(x) for x in header_parts])

                # Load Spectra Data
                absi = np.loadtxt(abs_f, skiprows=1)
                
                # Handle Time Columns
                if xi.ndim == 2 and xi.shape[1] > 1:
                    xi = xi[:, 1:] # Drop time col from conc file
                elif xi.ndim == 1:
                    xi = xi.reshape(-1, 1)

                # Absorbance Data: Drop time col (col 0)
                # Ensure we have data
                if absi.ndim == 1: absi = absi.reshape(1, -1)
                data_curr = absi[:, 1:] 
                
                if wavelengths is None:
                    wavelengths = wl_curr
                
                x_list.append(xi)
                absor_list.append(data_curr)
                
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")
        else:
            print(f"  [Skipping] Files not found: {x_f} or {abs_f}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        absor0 = np.vstack([wavelengths, absor_data])
        
        print(f"Total samples loaded: {x0.shape[0]}")
    else:
        print("No data loaded. Exiting.")
        return

    # Check dimensions
    if x0.shape[1] != nc:
        print(f"WARNING: Data has {x0.shape[1]} columns but nc={nc}. Adjusting nc.")
        nc = x0.shape[1]
        cname = cname[:nc]
        colors = colors[:nc]
    
    # =========================================================================
    #                                EXECUTION
    # =========================================================================
    
    engine = MulticalEngineSVR() # Using the new SVR Engine
    
    result = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
        frac_test, dadosteste, OptimModel, pretreat, 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier, use_ftest=use_ftest,
        colors=colors
    )

    if result is None:
        print("Engine run failed.")
        return

    # Unpacking 4 values (SVR Engine returns 4)
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc = result
    
    print("\nProcessing complete. Check 'results_svr_experiment/minimos.txt' for best parameters.")
    
    # Quick Plot of Grid Search Progress
    plt.figure(figsize=(10, 6))
    for j in range(nc):
        plt.plot(RMSECV_conc[:, j], label=f'{cname[j]} CV Error')
    plt.title('SVR Grid Search CV Error')
    plt.xlabel('Grid Index Reference')
    plt.ylabel(f'RMSE ({unid})')
    plt.legend()
    plt.grid(True)
    # plt.savefig(os.path.join(results_dir, 'grid_search_plot.png')) # Optional
    plt.show()

if __name__ == "__main__":
    main()
