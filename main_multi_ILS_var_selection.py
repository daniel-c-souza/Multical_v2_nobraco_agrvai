import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.var_selection import run_pso, run_simulated_annealing, calculate_vip, calculate_rmsecv_fast, select_k_ftest

def main():
    # =========================================================================
    #                               CONFIGURATION
    # =========================================================================
    
    # --- 1. General Settings ---
    results_dir = "results_var_selection"
    os.makedirs(results_dir, exist_ok=True)
    
    # --- 2. Model Parameters ---
    Selecao = 1       # Model type: 1 = PLS (Required for this variable selection logic)
    kmax = 15         # Max Latent Variables
    
    # Analytes
    nc = 3
    cname = ['cb', 'gl', 'xy'] 
    colors = ['green', 'red', 'purple']
    unid = 'g/L' 

    # Optimization Parameters (SPA/PCR specifics)
    optkini = 2 
    lini = 0 
    
    # --- 3. Cross-Validation Settings ---
    kpart = 5
    cv_type = 'venetian' 
    OptimModel = ['kfold', kpart, cv_type] 
    
    frac_test = 0.0 
    dadosteste = [] 

    # --- 4. Variable Selection Settings ---
    
    # Method Selection: 'VIP' or 'SA' (Simulated Annealing) or 'PSO'
    selection_method = 'VIP' 
    
    # VIP Parameters
    vip_thresholds = np.arange(0.5, 1.6, 0.1) # Thresholds to scan
    
    # Simulated Annealing Parameters
    sa_params = {
        'max_iter': 3000,     # Maximum number of iterations
        'initial_temp': 0.7, # Initial temperature (T0)
        'alpha': 0.92        # Cooling rate (T = T * alpha)
    }

    # PSO Parameters
    pso_params = {
        'n_particles': 100,
        'max_iter': 1000,
        'w': 0.9,   # Inertia weight
        'c1': 1.49, # Cognitive weight
        'c2': 1.49  # Social weight
    }

    # --- 5. Statistical Analysis ---
    outlier = 0 
    use_ftest = True 
    analysis_list = [['LB'], ['PCA']]

    # --- 6. Pretreatment Pipeline ---
    pretreat = [
        ['Cut', 4500, 8000, 1],
        ['SG', 7, 1, 2, 1, 1],
    ]

    # --- 7. Data Files ---
    data_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),   
        ('exp7_refe.txt', 'exp7_nonda.txt'),      
    ]

    # =========================================================================
    #                       DATA LOADING
    # =========================================================================

    x_list = []
    absor_list = []
    time_list_conc = []
    time_list_spec = []
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
                    ti_conc = xi[:, 0]
                    xi = xi[:, 1:] 
                else:
                    ti_conc = None
                    if xi.ndim == 1:
                        xi = xi.reshape(-1, 1)
                
                data_curr = absi[:, 1:] 
                ti_spec = absi[:, 0] 
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
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
        print(f"Total samples loaded: {x0.shape[0]}")
    else:
        raise FileNotFoundError("No valid data files loaded.")
    
    # --- EXPLICIT PRETREATMENT STEP ---
    # We apply pretreatment NOW so we can use the clean data for variable selection
    print("\n--- Applying Pretreatment for Variable Selection ---")
    
    # apply_pretreatment expects (absor, lambda) and returns transformed
    # We use a temp Absor0 to pass to apply_pretreatment, but we manually call it below
    
    absor_pre, wavelengths_pre = apply_pretreatment(pretreat, absor_data, wavelengths, output_dir=results_dir, prefix="VS_")
    
    if absor_pre.shape[0] != x0.shape[0]:
        print("Warning: Pretreatment changed number of samples? (Unexpected)")
    
    num_wavelengths = absor_pre.shape[1]
    
    # =========================================================================
    #                            VARIABLE SELECTION
    # =========================================================================
    
    print("\n=========================================================================")
    print(f"                    VARIABLE SELECTION OPTIMIZATION ({selection_method})")
    print("=========================================================================")
    print(f"Initial variables: {num_wavelengths}")
    
    best_mask = np.ones(num_wavelengths, dtype=bool)
    best_rmse_sel = 0.0
    
    if selection_method == 'VIP':
        # 1. Determine optimal K using all variables
        print("Step 1: Determining optimal LV (k) with all variables...")
        rmsecv_all, k_min, rmsecv_vector_all = calculate_rmsecv_fast(absor_pre, x0, kmax, folds=kpart, cv_type=cv_type)
        
        # Use F-test to select K for VIP Calculation
        k_opt_ftest = select_k_ftest(rmsecv_vector_all, n_cal=x0.shape[0])
        
        print(f" -> Best k (RMSECV minimized): {k_min} (RMSECV: {rmsecv_all:.4f})")
        print(f" -> Best k (F-test): {k_opt_ftest} (Used for VIP)")
        
        model_vip = PLS()
        
        # 2. Calculate VIP Scores based on optimal K (F-test selected)
        print("Step 2: Calculating VIP scores...")
        vip_scores = calculate_vip(model_vip, absor_pre, x0, k_opt_ftest, wavelengths=wavelengths_pre, output_dir=results_dir)
        
        # 3. Optimize VIP Threshold
        print("Step 3: Optimizing VIP threshold...")
        # Test thresholds from config
        thresholds = vip_thresholds
        
        best_thresh = 0.0
        best_rmse_sel = 1e9
        
        results_optimization = []
        
        for th in thresholds:
            mask = vip_scores >= th
            n_sel = np.sum(mask)
            
            if n_sel < 2: # Need at least some variables
                 continue
                 
            absor_sub = absor_pre[:, mask]
            rmse_sub, k_sub, _ = calculate_rmsecv_fast(absor_sub, x0, kmax, folds=kpart, cv_type=cv_type)
            
            results_optimization.append((th, n_sel, k_sub, rmse_sub))
            print(f"  Th={th:.1f}: {n_sel} vars, k={k_sub}, RMSECV={rmse_sub:.4f}")
            
            if rmse_sub < best_rmse_sel:
                best_rmse_sel = rmse_sub
                best_thresh = th
                best_mask = mask
        
        # Save Optimization Results
        with open(os.path.join(results_dir, "VIP_Optimization_Log.txt"), "w") as f:
            f.write("Threshold\tNumVars\tK_opt\tRMSECV\n")
            f.write(f"ALL\t{num_wavelengths}\t{k_min}\t{rmsecv_all:.4f}\n")
            for r in results_optimization:
                 f.write(f"{r[0]:.1f}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\n")
                 
        print(f"\nOptimization Finished.")
        print(f"Selected Threshold: {best_thresh:.1f}")
        
    elif selection_method == 'SA':
        print("Running Simulated Annealing...")
        # Note: You can tune max_iter, temp, alpha here
        best_mask, best_rmse_sel = run_simulated_annealing(absor_pre, x0, kmax, 
                                                         folds=kpart, cv_type=cv_type, 
                                                         max_iter=sa_params['max_iter'], 
                                                         initial_temp=sa_params['initial_temp'], 
                                                         alpha=sa_params['alpha'])

    elif selection_method == 'PSO':
        print("Running Particle Swarm Optimization (PSO)...")
        best_mask, best_rmse_sel = run_pso(absor_pre, x0, kmax, 
                                           folds=kpart, cv_type=cv_type, 
                                           n_particles=pso_params['n_particles'], 
                                           max_iter=pso_params['max_iter'], 
                                           w=pso_params['w'], 
                                           c1=pso_params['c1'], 
                                           c2=pso_params['c2'])

    else:
        print(f"Unknown Method {selection_method}. Selecting All.")
        best_mask = np.ones(num_wavelengths, dtype=bool)

    print(f"Selected {np.sum(best_mask)} / {num_wavelengths} variables.")
    # Calculate RMSECV for final selection if not already
    if selection_method != 'SA' and selection_method != 'VIP':
         rmse_final, _, _ = calculate_rmsecv_fast(absor_pre[:, best_mask], x0, kmax, folds=kpart, cv_type=cv_type)
         print(f"Estimated RMSECV: {rmse_final:.4f}")
    else:
         print(f"Estimated RMSECV: {best_rmse_sel:.4f}")

    # Plot Selected Variables on Mean Spectrum
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_spec = np.mean(absor_pre, axis=0)
    ax.plot(wavelengths_pre, mean_spec, 'k-', alpha=0.3, label='Mean Spectrum')
    ax.scatter(wavelengths_pre[best_mask], mean_spec[best_mask], c='r', s=5, label='Selected Vars')
    ax.set_title(f"Selected Variables ({selection_method})")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Absorbance")
    ax.legend()
    plt.savefig(os.path.join(results_dir, "Selected_Variables.png"))
    plt.show(block=False)

    
    # =========================================================================
    #                          FINAL EXECUTION
    # =========================================================================
    
    print("\n--- Running Final Calibration with Selected Variables ---")
    
    # Filter Data
    absor_final = absor_pre[:, best_mask]
    wavelengths_final = wavelengths_pre[best_mask]
    
    # Reconstruct inputs for engine
    # absor0 for engine expects Row 0 as Wavelengths
    absor0_final = np.vstack([wavelengths_final, absor_final])
    
    # Create Engine
    engine = MulticalEngine()
    
    # Run Engine
    # IMPORTANT: We pass pretreat_list=[] because we ALREADY applied pretreatment manually above.
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0_final, 
        frac_test, dadosteste, OptimModel, pretreat_list=[], 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier, use_ftest=use_ftest, colors=colors
    )
    
    if RMSECV is not None:
        print("\nFinal Results Available.")

    print("\nProcessing complete. Showing plots...")
    plt.show()

if __name__ == "__main__":
    main()
