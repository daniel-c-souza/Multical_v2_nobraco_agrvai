import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.models.pls import PLS
from src.multical.var_selection import run_pso, run_simulated_annealing, calculate_vip, calculate_rmsecv_fast, select_k_ftest

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. Selection Method ---
SELECTION_METHOD = 'VIP' # Options: 'VIP', 'SA' (Simulated Annealing), 'PSO' (Particle Swarm)

# --- 2. Output & Data ---
RESULTS_DIR = "results_var_selection"

DATA_FILES = [
    ('data/exp4_refe.txt', 'data/exp4_nonda.txt'),
    ('data/exp5_refe.txt', 'data/exp5_nonda.txt'),
    ('data/exp6_refe.txt', 'data/exp6_nonda.txt'),
    ('data/exp7_refe.txt', 'data/exp7_nonda.txt'),  
]

# --- 3. Model Parameters ---
MODEL_TYPE = 1          # 1 = PLS (Required for variable selection)
MAX_LATENT_VARS = 15    # Maximum Latent Variables
ANALYTES = ['cb', 'gl', 'xy']
UNITS = 'g/L'
COLORS = ['green', 'red', 'purple']

# Cross-Validation Settings
K_FOLDS = 5
CV_TYPE = 'venetian'

# --- 4. Optimization Parameters ---

# A) VIP Settings
VIP_THRESHOLDS = np.arange(0.1, 1.9, 0.1) # Thresholds to scan (0.5 to 1.5)

# B) Simulated Annealing (SA) Settings
SA_PARAMS = {
    'max_iter': 3000,     # Maximum iterations
    'initial_temp': 0.7, # Initial temperature
    'alpha': 0.92        # Cooling rate
}

# C) PSO Settings
PSO_PARAMS = {
    'n_particles': 100,
    'max_iter': 1000,
    'w': 0.9,   # Inertia weight
    'c1': 1.49, # Cognitive weight
    'c2': 1.49  # Social weight
}

# --- 5. Pretreatment Pipeline ---
# Applied BEFORE selection logic
PRETREATMENT = [
    ['Cut', 4400, 7500, 1],
    ['SG', 7, 2, 1, 1],
]

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def load_data(files):
    x_list, absor_list, wavelengths = [], [], None
    print("Loading data...")
    for x_f, abs_f in files:
        if not (os.path.exists(x_f) and os.path.exists(abs_f)):
            print(f"Skipping {x_f}: Not found.")
            continue
            
        try:
            xi = np.loadtxt(x_f)
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:]
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])
            absi = np.loadtxt(abs_f, skiprows=1)[:, 1:]

            if wavelengths is None: 
                wavelengths = wl_curr
            elif len(wavelengths) == len(wl_curr) and not np.allclose(wavelengths, wl_curr, atol=1e-1):
                print(f"Warning: Wavelength mismatch.")

            x_list.append(xi)
            absor_list.append(absi)
        except Exception as e:
            print(f"Error loading {x_f}: {e}")

    if not x_list: return None, None, None
    
    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    return x0, absor_data, wavelengths

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Data
    x0, absor_data, wavelengths = load_data(DATA_FILES)
    if x0 is None: return

    # 2. Pretreatment
    print("\n--- Applying Pretreatment ---")
    # We apply passing a copy as absor_raw
    absor_pre, wavelengths_pre = apply_pretreatment(PRETREATMENT, absor_data, wavelengths, output_dir=RESULTS_DIR, prefix="VS_")
    
    # 3. Variable Selection
    print(f"\n--- Running Variable Selection: {SELECTION_METHOD} ---")
    best_mask = np.ones(len(wavelengths_pre), dtype=bool) # Default is All

    if SELECTION_METHOD == 'VIP':
        # Step 1: Find optimal K with all variables
        print(" Determining optimal K (Full Spectrum)...")
        rmse_all, k_min, rmsecv_vec = calculate_rmsecv_fast(absor_pre, x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
        
        # Prefer F-test k_min if available, else plain min
        k_opt_ftest = select_k_ftest(rmsecv_vec, n_cal=x0.shape[0])
        print(f" Optimal K: {k_opt_ftest} (RMSE={rmse_all:.4f})")

        # Step 2: Calculate VIP Scores
        print(" Calculating VIP scores...")
        model_vip = PLS()
        vip_scores = calculate_vip(model_vip, absor_pre, x0, k_opt_ftest, wavelengths=wavelengths_pre, output_dir=RESULTS_DIR)
        
        # Step 3: Optimize Threshold
        print(" Optimizing VIP Threshold...")
        best_rmse = 1e9
        best_thresh = 0.0
        
        for th in VIP_THRESHOLDS:
            mask = vip_scores >= th
            n_sel = np.sum(mask)
            if n_sel < 2: continue # Too few variables
            
            rmse_sub, k_sub, _ = calculate_rmsecv_fast(absor_pre[:, mask], x0, MAX_LATENT_VARS, K_FOLDS, CV_TYPE)
            print(f"  Th={th:.1f}: {n_sel} variables | RMSECV: {rmse_sub:.4f} (k={k_sub})")
            
            if rmse_sub < best_rmse:
                best_rmse = rmse_sub
                best_thresh = th
                best_mask = mask
        
        print(f" Best Threshold: {best_thresh:.1f} (RMSE={best_rmse:.4f})")

    elif SELECTION_METHOD == 'SA':
        best_mask, _ = run_simulated_annealing(absor_pre, x0, MAX_LATENT_VARS, folds=K_FOLDS, cv_type=CV_TYPE, **SA_PARAMS)

    elif SELECTION_METHOD == 'PSO':
        best_mask, _ = run_pso(absor_pre, x0, MAX_LATENT_VARS, folds=K_FOLDS, cv_type=CV_TYPE, **PSO_PARAMS)

    else:
        print(" Unknown Method. Using all variables.")

    # 4. Visualization
    n_sel = np.sum(best_mask)
    print(f"\nSelected {n_sel} / {len(wavelengths_pre)} variables.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_spec = np.mean(absor_pre, axis=0)
    ax.plot(wavelengths_pre, mean_spec, 'k-', alpha=0.3, label='Mean Spectrum')
    ax.scatter(wavelengths_pre[best_mask], mean_spec[best_mask], c='r', s=5, label='Selected')
    ax.set_title(f"Selected Variables ({SELECTION_METHOD})")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "Selected_Variables.png"))
    # plt.show() # Uncomment to see interactive plot

    # --- Save Selected Wavelengths ---
    wavelengths_selected = wavelengths_pre[best_mask]
    save_path = os.path.join(RESULTS_DIR, "selected_wavelengths.txt")
    np.savetxt(save_path, wavelengths_selected, fmt='%.4f', header="Selected Wavelengths (cm-1)")
    print(f"Saved selected wavelengths to: {save_path}")

    # 5. Final Calibration
    print("\n--- Running Final Calibration with Selected Variables ---")
    engine = MulticalEngine()
    
    absor_final = absor_pre[:, best_mask]
    wl_final = wavelengths_pre[best_mask]
    
    # Engine expects row 0 as wavelengths
    absor0_final = np.vstack([wl_final, absor_final])
    
    OptimModel = ['kfold', K_FOLDS, CV_TYPE]
    nc = len(ANALYTES)
    
    # Run engine with PRETREATED data (so pass empty list for pretreat pipeline)
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc, R2CV, R2cal, best_k_dict = engine.run(
        MODEL_TYPE, 2, 0, MAX_LATENT_VARS, nc, ANALYTES, UNITS, 
        x0, absor0_final, 0.0, [], OptimModel, pretreat_list=[], 
        analysis_list=[['LB'], ['PCA']], output_dir=RESULTS_DIR, colors=COLORS
    )

    if RMSECV is not None:
        print("\nFinal Model Generated.")

    # --- SAVE SELECTION MODEL ---
    from src.multical.core.saving import train_and_save_model_pls
    
    print("\n--- Saving Variable Selection Model ---")
    MODEL_FILENAME = "model_var_selection.pkl"
    # Note: absor_final is already pretreated and reduced.
    # engine.run returns best_k_dict which is list of k
    train_and_save_model_pls(absor_final, x0, wl_final, best_k_dict, os.path.join(RESULTS_DIR, MODEL_FILENAME), rmsecv_list=RMSECV_conc)


if __name__ == "__main__":
    main()
