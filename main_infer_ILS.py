import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.inference import run_inference

def load_data_from_list(file_pairs):
    """
    Helper to load pairs of (Concentration, Absorbance) files.
    """
    x_list = []
    absor_list = []
    wavelengths = None
    
    for x_f, abs_f in file_pairs:
        if os.path.exists(x_f) and os.path.exists(abs_f):
            print(f"  - Loading {x_f} / {abs_f}")
            try:
                xi = np.loadtxt(x_f)
                absi = np.loadtxt(abs_f)
                
                # Absorbance file has wavelengths in first row
                wl_curr = absi[0, :]
                data_curr = absi[1:, :]
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
                    # Checking consistency
                    if not np.allclose(wavelengths, wl_curr, atol=1e-1):
                         print(f"Warning: Wavelengths in {abs_f} differ from previous.")
                
                x_list.append(xi)
                absor_list.append(data_curr)
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")
        else:
            print(f"  [Skipping] Files not found: {x_f} or {abs_f}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        # Combine: Row 0 is wavelengths
        absor0 = np.vstack([wavelengths, absor_data])
        return x0, absor0
    else:
        return None, None

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
    kinf = [4, 4, 4] 
    
    # Analytes
    nc = 3
    cname = ['clb', 'gl', 'xyl'] 
    unid = 'g/L'
    
    # =========================================================================
    #                                   DATA
    # =========================================================================
    
    # --- Calibration Data ---
    # Files to build the model
    cal_files = [
        ('x02.txt', 'abs02.txt'),
        #('x03.txt', 'abs03.txt'),
        # ('x04.txt', 'abs04.txt') 
    ]
    
    # --- Inference Data ---
    # Files to predict on
    inf_files = [
        ('x04.txt', 'abs04.txt'), # Using x04 as inference for demo
        # ('x05.txt', 'abs05.txt')
    ]
    
    print("Loading Calibration Data...")
    x0, absor0 = load_data_from_list(cal_files)
    if x0 is None:
        print("Error: No calibration data loaded.")
        return

    print("Loading Inference Data...")
    xinf0, absorinf0 = load_data_from_list(inf_files)
    if absorinf0 is None:
        print("Error: No inference data loaded.")
        return

    # =========================================================================
    #                                PRETREATMENT
    # =========================================================================
    
    # Pretreatment for Calibration
    pretreat = [
        ['Cut', 4500, 10000, 1],
        ['SG', 3, 1, 5, 1, 1],
        ['MA', 7, 1, 1],
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
    
    run_inference(
        Selecao, optkini, lini, kinf, nc, cname, unid, 
        x0, absor0, xinf0, absorinf0, 
        pretreat, pretreatinf, 
        analysis_list=analysis_list, analysisinf_list=analysisinf_list, 
        leverage=leverage, output_dir=results_dir
    )
    
    print("\nProcessing complete. Showing plots...")
    plt.show()

if __name__ == "__main__":
    main()
