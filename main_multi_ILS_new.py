import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.analysis import func_analysis

def main():
    # =========================================================================
    #                                   CONFIG
    # =========================================================================
    
    # Output Directory
    # Directory to save text results and plots
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Method
    Selecao = 1 # 1 = PLS; 2 = SPA; 3=PCR
    optkini = 2 # 0=> lini = lambda(1); 1=> lini = dado abaixo; 2=> otimiza lini.
    lini = 0 # [nm] só usar se optkini=1.

    # Analysis Configuration
    # Define which analyses to run.
    # Format: list of lists, e.g., [['LB'], ['PCA']]
    # Options: 'LB' (Lambert-Beer plots), 'PCA' (Principal Component Analysis)
    analysis_list = [['LB'], ['PCA']]
    
    # Max Regressors
    kmax =20
    
    # Num Analytes
    nc = 3
    
    cname = ['cb','gl','xy'] # constituent names
    unid = 'g/L' #  unit for constituents
    
    # =========================================================================
    #                                   DATA
    # =========================================================================
    
    # Define file pairs to load. 
    # Each entry is (Concentration_File, Absorbance_File)
    
    # Example using specific files:
    data_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),
    ]
     
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
    #                                ANALYSIS
    # =========================================================================
    

    frac_test = 0.0 #random fraction for test set
    dadosteste = [] # Empty list or tuple if no external test set
    
    # Optimization of model complexity parameters
    # Option 1: Hold-out Validation with frac_val
    frac_val = 0.20 #0.05-0.4
    #OptimModel = ['Val', frac_val]
    
    # Option 2: K-Fold Cross Validation
    # Set kpart = -1 for LOOCV, or integer > 1 for k-folds
    kpart = 10
    # CV Type: 'random', 'consecutive', or 'venetian'
    cv_type = 'venetian'
    OptimModel = ['kfold', kpart, cv_type] 
    
    # Option 3: Integer (Legacy support for k-fold)
    # OptimModel = -1 # -1 = LOOCV
    
    # Outlier handling
    outlier = 0 # 0=No, 1=Yes (Calculates Student t-test on residuals)

  
    
    # =========================================================================
    #                                PRETREATMENT
    # =========================================================================
    
    """ Set of pretreatment 
     Moving Average: ['MA',radius,Losing points = 1 or 2, plot=1 or 0]
     LOESS: ['Loess',alpha = [0.2-0.9], order = [1,2,...],plot=1 or 0]
     Savitzky and Golay Filter: ['SG',radius,Losing points = 1 or 2,poly order = integer, der order = integer, plot=1 or 0]   
     Derivative: ['Deriv',order,plot=1 or 0]
     Cutting regions: ['Cut',lower bound,upper bound,plot=1 or 0]
     Cutting maxAbs: ['CutAbs',maxAbs, action: warning = 0 or cutting = 1 ,plot=1 or 0]
     Control Baseline based on noninformation region: ['BLCtr',ini_lamb,final_lamb,Abs_value, plot=1 or 0]
    """      

    pretreat = [
        ['Cut', 5500, 8500, 1],
        ['SG',7,1,2,1,1],
        
    ]
    
    
    # =========================================================================
    #                                EXECUTION
    # =========================================================================
    
    engine = MulticalEngine()
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
        frac_test, dadosteste, OptimModel, pretreat, 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier
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
