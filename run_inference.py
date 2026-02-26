import numpy as np
import os
import matplotlib.pyplot as plt
from src.multical.inference import run_inference

# =============================================================================
#                                 CONFIGURATION
# =============================================================================

# --- 1. General Settings ---
RESULTS_DIR = "results_inference" # Directory for output

# --- 2. Data Files ---
# Calibration Data: Used to build the model (reference)
CALIBRATION_FILES = [
    ('data/exp4_refe.txt', 'data/exp4_nonda.txt'),
    ('data/exp5_refe.txt', 'data/exp5_nonda.txt'),
    ('data/exp6_refe.txt', 'data/exp6_nonda.txt'),
    #('data/exp7_refe.txt', 'data/exp7_nonda.txt'),
]

# Inference Data: New data to predict
# Format: (Reference_File_Optional, Absorbance_Spectra_File)
# If reference file exists, it will be used for validation plots.
INFERENCE_FILES = [
    #('data/exp4_refe.txt', 'data/exp4_nonda.txt'),
    #('data/exp5_refe.txt', 'data/exp5_nonda.txt'),
    #('data/exp6_refe.txt', 'data/exp6_nonda.txt'),
    ('data/exp7_refe.txt', 'data/exp7_nonda.txt'),
]

# --- 3. Model Parameters ---
MODEL_TYPE = 1          # 1=PLS, 2=SPA, 3=PCR

# Important: These must match the training settings or be optimized
# Number of Latent Variables (LVs) for each analyte [cb, gl, xy]
K_VARS_PER_ANALYTE = [5, 6, 6] 

ANALYTES = ['cb', 'gl', 'xy']
COLORS = ['green', 'red', 'purple']
UNITS = 'g/L'

# --- 4. Pretreatment Pipeline ---
# Must match the calibration model exactly!
PRETREATMENT = [
    ['Cut', 4400, 8500, 0],
    ['SG', 7, 1, 2, 1, 0],
]

# Analysis options for diagnostic plots
ANALYSIS_LIST = [] 
LEVERAGE = 0 # Calculate leverage? (0=No)

# =============================================================================
#                              MAIN EXECUTION
# =============================================================================

def load_calibration_data(files):
    x_list, absor_list, wavelengths = [], [], None
    print("Loading calibration data...")
    for x_f, abs_f in files:
        if not (os.path.exists(x_f) and os.path.exists(abs_f)): continue
        try:
            xi = np.loadtxt(x_f)
            if xi.ndim == 2 and xi.shape[1] > 1: xi = xi[:, 1:]
            if xi.ndim == 1: xi = xi.reshape(-1, 1)

            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            wl_curr = np.array([float(x) for x in header[1:]])
            absi = np.loadtxt(abs_f, skiprows=1)[:, 1:]

            if wavelengths is None: wavelengths = wl_curr
            x_list.append(xi)
            absor_list.append(absi)
        except Exception as e:
            print(f"Error loading {x_f}: {e}")
            
    if not x_list: return None, None, None
    x0 = np.vstack(x_list)
    absor_data = np.vstack(absor_list)
    absor0 = np.vstack([wavelengths, absor_data])
    return x0, absor0, None

def load_inference_data(files, nc):
    x_list, absor_list, sizes, wavelengths = [], [], [], None
    print("Loading inference data...")
    
    for x_f, abs_f in files:
        if not os.path.exists(abs_f): continue
        try:
            # Load Spectra
            with open(abs_f, 'r') as f: header = f.readline().strip().split()
            # Try to handle "Time" or "Wavenumber" headers
            start_idx = 1
            wl = np.array([float(x) for x in header[start_idx:]])
                
            absi = np.loadtxt(abs_f, skiprows=1)[:, 1:]
            if wavelengths is None: wavelengths = wl
            
            n_samples = absi.shape[0]
            sizes.append(n_samples)
            absor_list.append(absi)
            
            # Load Reference (if available) and align
            xi_aligned = np.full((n_samples, nc), np.nan)
            if os.path.exists(x_f):
                try:
                    ref_data = np.loadtxt(x_f)
                    # Helper logic: Check basic shape
                    if ref_data.ndim > 1 and ref_data.shape[1] >= nc + 1:
                        # Assuming ref_data is [Time, Val1, Val2, Val3]
                        # And we want to match times roughly? 
                        # For simple cases, we might just trim if sizes differ slightly
                        # or fill what we can.
                        n_ref = ref_data.shape[0]
                        limit = min(n_samples, n_ref)
                        xi_aligned[:limit, :] = ref_data[:limit, 1:nc+1]
                except Exception as e:
                    print(f"Warning loading ref {x_f}: {e}")
            
            x_list.append(xi_aligned)

        except Exception as e:
            print(f"Error loading {abs_f}: {e}")

    if not x_list: return None, None, None, []
    
    xinf0 = np.vstack(x_list)
    absorinf_data = np.vstack(absor_list)
    absorinf0 = np.vstack([wavelengths, absorinf_data])
    
    return xinf0, absorinf0, None, sizes

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams['figure.max_open_warning'] = 100

    # 1. Load Data
    x0, absor0, t0 = load_calibration_data(CALIBRATION_FILES)
    if x0 is None: return

    xinf0, absorinf0, tinf0, inf_sizes = load_inference_data(INFERENCE_FILES, len(ANALYTES))
    if absorinf0 is None: return

    # 2. Run Inference
    print("\nRunning Inference...")
    
    # Run the engine
    X_pred, rmsecv = run_inference(
        MODEL_TYPE, 2, 0, K_VARS_PER_ANALYTE, len(ANALYTES), ANALYTES, UNITS, 
        x0, absor0, xinf0, absorinf0, 
        PRETREATMENT, PRETREATMENT, 
        analysis_list=ANALYSIS_LIST, analysisinf_list=[], 
        leverage=LEVERAGE, output_dir=RESULTS_DIR, colors=COLORS
    )
    
    if X_pred is not None:
        print(f"\nInference Complete. Predictions shape: {X_pred.shape}")
        print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
