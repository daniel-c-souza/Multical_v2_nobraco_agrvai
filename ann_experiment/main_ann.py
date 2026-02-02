import numpy as np
import os
import sys
import time

# Add project root to sys.path to allow imports from src and ann_experiment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
from ann_experiment.ann_engine import MulticalEngine
from src.multical.analysis import func_analysis
plt.rcParams['figure.max_open_warning'] = 100

def main():
    # =========================================================================
    #                               CONFIGURATION
    # =========================================================================
    
    # --- 1. General Settings ---
    results_dir = "results_ann_experiment"  # Directory to save results
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. Model Parameters ---
    Selecao = 5       # Model type: 5 = ANN (PyTorch)
    kmax = 2000       # Max combinations (Overnight run setting)
    
    # Analytes (Constituents)
    nc = 3            # Number of analytes
    cname = ['cb', 'gl', 'xy']  # Names for output checks
    colors = ['green', 'red', 'purple'] 
    unid = 'g/L'      

    # Optimization (Legacy)
    optkini = 2       
    lini = 0          

    # --- 3. Cross-Validation & Validation Settings ---
    frac_test = 0.0   
    dadosteste = []   
    
    validation_mode = 'kfold' 
    kpart = 5                 
    cv_type = 'venetian'      
    frac_val = 0.20           
    
    if validation_mode == 'kfold':
        OptimModel = ['kfold', kpart, cv_type]
    elif validation_mode == 'Val':
        OptimModel = ['Val', frac_val]
    else:
         OptimModel = ['kfold', 5, 'venetian']

    # --- 4. Statistical Analysis ---
    outlier = 0       
    use_ftest = False  # Not used for ANN
    
    analysis_list = []

    # --- 5. Data Files ---
    data_files = [ 
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),   
        ('exp7_refe.txt', 'exp7_nonda.txt'),
    ]

    # --- 6. Pretreatment Pipeline ---
    pretreat = [
        ['Cut', 4500, 8000, 1],
        #['SG', 7, 1, 2, 1, 1],
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
                xi = np.loadtxt(x_f)
                with open(abs_f, 'r') as f_node:
                    header_parts = f_node.readline().strip().split()
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                absi = np.loadtxt(abs_f, skiprows=1)
                
                if xi.ndim == 2 and xi.shape[1] > 1:
                    xi = xi[:, 1:] 
                else:
                    if xi.ndim == 1: xi = xi.reshape(-1, 1)
                
                data_curr = absi[:, 1:] 
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
                    if not np.allclose(wavelengths, wl_curr, atol=1e-1):
                         print(f"Warning: Wavelengths in {abs_f} differ.")
                
                x_list.append(xi)
                absor_list.append(data_curr)
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        absor0 = np.vstack([wavelengths, absor_data])
        print(f"Total samples loaded: {x0.shape[0]}")
    
    print(f"DEBUG: nc={nc}, x0.shape={x0.shape}")
    if x0.shape[1] != nc:
        print(f"WARNING: Data has {x0.shape[1]} columns but nc={nc}. Adjusting nc to match data.")
        nc = x0.shape[1]
        cname = cname[:nc]
        colors = colors[:nc]
    
    # =========================================================================
    #                                EXECUTION
    # =========================================================================
    
    engine = MulticalEngine()
    
    start_time = time.time()
    result = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
        frac_test, dadosteste, OptimModel, pretreat, 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier, use_ftest=use_ftest,
        colors=colors
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")

    if result is not None and result[0] is not None:
        # result[0] is RMSECV, shape (num_models, nc)
        num_models_run = result[0].shape[0]
        if num_models_run > 0:
            avg_time = elapsed_time / num_models_run
            print(f"Average time per architecture: {avg_time:.2f} seconds")
            print(f"Estimated time for 50 architectures: {avg_time * 50:.2f} seconds ({avg_time * 50 / 60:.2f} minutes)")
            print(f"Estimated time for 100 architectures: {avg_time * 100:.2f} seconds ({avg_time * 100 / 60:.2f} minutes)")

    if result is None or result[0] is None:
        print("Engine run failed.")
        return

    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = result
    
    print(f"\nResults for Model: ANN (PyTorch)")
    print("RMSECVconc (Original Units):")
    print(RMSECV_conc)
    
    print("\nProcessing complete. Plots saved to output directory.")
    # plt.show()

if __name__ == "__main__":
    main()
