import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.core.engine import MulticalEngine
from src.multical.preprocessing.pipeline import apply_pretreatment

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
RESULTS_DIR = "results"         # Directory where results and plots will be saved

# --- 2. Data Files ---
# List of (Concentration_File, Absorbance_File)
# Ensure these are in your 'data/' folder
DATA_FILES = [
    ('data/exp4_refe.txt', 'data/exp4_nonda.txt'),
    ('data/exp5_refe.txt', 'data/exp5_nonda.txt'),
    ('data/exp6_refe.txt', 'data/exp6_nonda.txt'),
    #('data/exp7_refe.txt', 'data/exp7_nonda.txt'),
]

# --- 3. Model Parameters ---
MODEL_TYPE = 1          # 1 = PLS (Partial Least Squares)
                        # 2 = SPA (Successive Projections Algorithm)
                        # 3 = PCR (Principal Component Regression)

MAX_LATENT_VARS = 15    # Maximum number of Latent Variables (Factors) to test
ANALYTES = ['cb', 'gl', 'xy']  # Names of the analytes (e.g. ['Glucose', 'Biomass'])
COLORS = ['green', 'red', 'purple'] # Plot colors for each analyte
UNITS = 'g/L'           # Unit of measurement

# SPA/PCR Specifics (Advanced)
SPA_OPT_K_INI = 2       # 0=> lini = lambda(1); 1=> lini = below; 2=> optimize lini.
SPA_L_INI = 0           # [cm-1] Initial wavenumber (only if optkini=1)

# --- 4. Cross-Validation & Validation Settings ---
# Test Set Split
TEST_SPLIT = 0.0          # Fraction of data to keep as a pure Test Set (0.0 to 1.0)
MANUAL_TEST_SET = []      # Manual test set (X_test, Y_test) if available, else []

# Cross-Validation
VALIDATION_MODE = 'kfold' # 'kfold' or 'Val' (Holdout)

# If 'kfold':
K_FOLDS = 5               # Number of folds for CV
CV_TYPE = 'venetian'      # Type of CV: 'random', 'consecutive', 'venetian'

# If 'Val' (Holdout):
VAL_FRACTION = 0.20       # Fraction for validation holdout

# --- 5. Pretreatment Pipeline ---
# Text-based pipeline definition.
# [Operation, Param1, Param2, ...]
PRETREATMENT = [
    ['Cut', 4400, 7500, 1], # Cut spectral region (Min, Max, Plot?)
    ['SG', 7, 1, 2, 1, 1],  # Savitzky-Golay: Window=7, Poly=1, Deriv=2

]

# --- 6. Analysis Settings ---
OUTLIER_REMOVAL = 0     # 0 = Off, 1 = On (Student t-test on residuals)
USE_F_TEST = True       # Use Osten F-test for automatic model selection (Optimal k)
PRE_ANALYSIS = [['LB'], ['PCA']] # Analyses to run before calibration

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def load_data(files):
    """
    Loads concentration and absorbance files into single matrices.
    """
    x_list, absor_list, wavelengths = [], [], None
    
    print("Loading data...")
    for x_f, abs_f in files:
        if not (os.path.exists(x_f) and os.path.exists(abs_f)):
            print(f"  [Skipping] File not found: {x_f} or {abs_f}")
            continue

        print(f"  - {x_f} / {abs_f}")
        try:
            # Load Concentration
            xi = np.loadtxt(x_f)
            # Handle dimensions
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:] # Drop time col if present
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            # Load Spectra (with header parsing)
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])
            
            # Load data skipping header
            absi = np.loadtxt(abs_f, skiprows=1)
            absi_data = absi[:, 1:] # Drop time column (col 0)

            if wavelengths is None: 
                wavelengths = wl_curr
            elif len(wavelengths) == len(wl_curr) and not np.allclose(wavelengths, wl_curr, atol=1e-1):
                print(f"Warning: Wavelength mismatch in {abs_f}")

            x_list.append(xi)
            absor_list.append(absi_data)
        except Exception as e:
            print(f"Error loading {x_f}: {e}")

    if not x_list: 
        raise FileNotFoundError("No valid data loaded. Check file paths.")

    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    
    # Engine expects wavelengths as the first row of absorbance matrix
    absor0 = np.vstack([wavelengths, absor_data]) 
    
    return x0, absor0, wavelengths, absor_data

def main():
    # --- Setup Styling ---
    plt.rcParams['figure.max_open_warning'] = 100
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load Data ---
    x0, absor0, wavelengths, absor_raw = load_data(DATA_FILES)
    print(f"Total samples: {x0.shape[0]}, Wavelengths: {len(wavelengths)}")

    # --- Plot Raw vs Pretreated ---
    print("\nGenerating Plots...")
    
    # 1. Raw
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, absor_raw.T)
    ax.set_title("a) Raw Spectra")
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Absorbance")
    fig.savefig(os.path.join(RESULTS_DIR, "Spectra_Raw.png"))
    plt.close(fig)
    
    # 2. Pretreated (Visualization only)
    absor_pre, wl_pre = apply_pretreatment(PRETREATMENT, absor_raw.copy(), wavelengths.copy(), plot=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(wl_pre, absor_pre.T)
    ax2.set_title("b) Pretreated Spectra")
    ax2.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax2.set_ylabel("Absorbance")
    fig2.savefig(os.path.join(RESULTS_DIR, "Spectra_Pretreated.png"))
    plt.close(fig2)

    # --- Run Multical Engine ---
    print("\nRunning Calibration Engine...")
    
    # Construct OptimModel
    if VALIDATION_MODE == 'kfold':
        OptimModel = ['kfold', K_FOLDS, CV_TYPE]
    elif VALIDATION_MODE == 'Val':
        OptimModel = ['Val', VAL_FRACTION]
    else:
        OptimModel = ['kfold', 5, 'venetian']

    nc = len(ANALYTES)
    engine = MulticalEngine()
    
    # Run
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc, R2CV, R2cal, best_k_dict = engine.run(
        MODEL_TYPE, SPA_OPT_K_INI, SPA_L_INI, MAX_LATENT_VARS, nc, ANALYTES, UNITS, 
        x0, absor0, TEST_SPLIT, MANUAL_TEST_SET, OptimModel, PRETREATMENT, 
        analysis_list=PRE_ANALYSIS, output_dir=RESULTS_DIR, outlier=OUTLIER_REMOVAL, 
        use_ftest=USE_F_TEST, colors=COLORS
    )

    # --- Helper for cleaner output ---
    def print_metric_table(title, matrix, names):
        if matrix is None: return
        print(f"\n> {title}")
        header_str = f"{'LV':<4} |" + "".join([f"{name:^12} |" for name in names])
        div_line = "-" * len(header_str)
        print(div_line)
        print(header_str)
        print(div_line)
        for k, row in enumerate(matrix):
            row_str = f"{k+1:<4} |" + "".join([f"{val:^12.5f} |" for val in row])
            print(row_str)
        print(div_line)

    if RMSECV is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS REPORT".center(60))
        print(f"{'='*60}")
        print_metric_table(f"RMSECV ({UNITS})", RMSECV_conc, ANALYTES)
        print_metric_table("R-Squared (CV)", R2CV, ANALYTES)

if __name__ == "__main__":
    main()
