import numpy as np
import os
import sys

# Add parent directory to path to allow importing 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import glob
import matplotlib.pyplot as plt
import pandas as pd

from src.multical.preprocessing.pipeline import apply_pretreatment

def load_inference_data(file_paths, output_dir=None):
    """
    Loads spectral data for inference.
    Supports CSV (standard) or legacy formats.
    Returns:
        X_inf: (n_samples, n_wavelengths)
        Wavelengths: (n_wavelengths)
        File_Names: List of source files
    """
    X_inf_list = []
    Wl_list = []
    Names_list = []
    
    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            continue
            
        try:
            print(f"Loading {fpath}...")
            # Try loading as standard CSV/TXT with headers
            # Check delimiter
            with open(fpath, 'r') as f:
                header = f.readline()
                sep = ';' if ';' in header else '\t'
                if ',' in header and sep == '\t': sep = ',' # Fallback
            
            # Simple loading - modify based on YOUR specific file format
            # Currently assuming format: Header has Wavelengths (or first col is Time/ID)
            # If files are like exp_04_inf.csv (Semicolon, transposed?)
            
            try:
                if 'exp_' in fpath and 'inf.csv' in fpath:
                     # Tall format: Col 0 = Wavenumber, Cols 1..N = Samples
                     df = pd.read_csv(fpath, sep=';', header=None)
                     # Wavelengths are Col 0
                     wl = df.iloc[:, 0].values
                     # Spectra are Cols 1..N
                     x = df.iloc[:, 1:].values.T # Transpose to (Samples, Waves)
                else:
                     # Wide format (Standard): Row 0=Header, Col 0=Index
                     # Use numpy for safety if pandas fails
                     data = np.loadtxt(fpath) # Automatically handles whitespace
                     # Check shape
                     if data.shape[1] > 2000: # Approx number of wavelengths
                          x = data
                          wl = np.arange(data.shape[1]) # Dummy if no header?
                          # Actually, we need wavelengths from somewhere.
                          # If loading 'exp4_nonda.txt', format is just a matrix.
                          # Wavelengths are usually consistent across project.
                     else:
                          # Assume Col 0 is wavelengths, Cols 1+ are samples?
                          wl = data[:, 0]
                          x = data[:, 1:].T
            except Exception:
                 # Last resort: Load with pandas, infer header
                 df = pd.read_csv(fpath, sep=sep)
                 x = df.values
                 wl = np.array([float(c) for c in df.columns if c.replace('.','',1).isdigit()])
                 if len(wl) == 0:
                      wl = None

            if x is not None:
                X_inf_list.append(x)
                if wl is not None:
                     Wl_list.append(wl)
                Names_list.append(os.path.basename(fpath))
                
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return X_inf_list, Wl_list, Names_list


def main_inference():
    # 1. Configuration
    model_path = "results_var_selection/final_model_pls.pkl"
    # List files to infer
    # Example: exp_04_inf.csv, etc.
    inference_files = glob.glob("exp_*_inf.csv") + glob.glob("exp*_nonda.txt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'main_multi_ILS_var_selection.py' first to train and save the model.")
        return

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    # Extract Parameters
    beta = model_data['Beta']
    x_med = model_data['Xmed']
    x_sig = model_data['Xsig']
    y_med = model_data['Ymed']
    y_sig = model_data['Ysig']
    mask = model_data['selected_mask']
    wl_model = model_data.get('wavelengths_model', None)
    wl_raw_model = model_data.get('wavelengths_raw', None)
    pretreat_config = model_data.get('pretreat_config', [])
    k_opt = model_data.get('k_opt', 0)
    
    print(f"Model loaded. LVs: {k_opt}, Selected Vars: {np.sum(mask)}")
    print(f"Pretreatment steps: {pretreat_config}")

    # 3. Process Each File
    for fpath in inference_files:
        print(f"\nProcessing {fpath}...")
        
        # A. Load raw data
        # Note: We need a robust loader that gets correctly formatted (Samples x Wavelengths) data
        # AND accurate wavelengths to match the model.
        # Here we re-use logic similar to training script for consistency if possible.
        
        # Quick Hack for this specific workspace structure:
        # If 'inf.csv' file:
        if 'inf.csv' in fpath:
             df = pd.read_csv(fpath, sep=';', header=None)
             wl_raw = df.iloc[:, 0].values
             x_raw = df.iloc[:, 1:].values.T
        elif 'nonda.txt' in fpath:
             # Try loading with header
             try:
                 with open(fpath, 'r') as f:
                     header_line = f.readline().strip().split()
                     # Assume first col is Time if header has text, otherwise assume raw
                     # Check if first element is number
                     try:
                         float(header_line[0])
                         has_text_header = False
                     except ValueError:
                         has_text_header = True
                         
                 if has_text_header:
                     # Standard nonda format: Header row with wavelengths (col 0 = Time/ID)
                     # Load skipping header
                     data_block = np.loadtxt(fpath, skiprows=1)
                     # Wavelengths from header (skip col 0)
                     wl_raw = np.array([float(x) for x in header_line[1:]])
                     x_raw = data_block[:, 1:]
                 else:
                     # No header? 
                     data_block = np.loadtxt(fpath)
                     x_raw = data_block
                     wl_raw = None # Fallback to model's raw wavelengths
             except Exception as e:
                 print(f"Error reading header of {fpath}: {e}")
                 continue
        else:
             continue # Skip unrecognized
             
        # B. Check Dimensions
        if wl_raw is not None and wl_raw_model is not None:
             # Check if wavelengths match model's expected input
             if len(wl_raw) != len(wl_raw_model):
                  print(f"Warning: Wavelength count mismatch ({len(wl_raw)} vs {len(wl_raw_model)}).")
                  # Can't easily fix without interpolation.
        
        # C. Pretreatment
        # If we didn't load wavelengths, fallback to model's raw wavelengths
        
        current_wl = wl_raw if wl_raw is not None else wl_raw_model
        
        if current_wl is None:
             print("Error: No wavelength information available for pretreatment. Skipping file.")
             continue

        # Debug print
        print(f"  Data shape: {x_raw.shape}, Wavelengths: {len(current_wl)}")
             
        x_pre, wl_pre = apply_pretreatment(pretreat_config, x_raw, current_wl, plot=False) # plot=False for batch mode

        # D. Variable Selection
        # The model expects input dimensions equal to the SELECTED variables.
        # So we must index `x_pre` using `mask`.

        
        # Validation:
        if x_pre.shape[1] != len(mask):
             print(f"Error: Pretreated data has {x_pre.shape[1]} vars, but mask expects {len(mask)}.")
             print("Skipping dimensions check/mask application if dimensions mismatch drastically.")
             # If already filtered (unlikely), proceed.
        
        x_sel = x_pre[:, mask]
        
        # E. Prediction
        # Y_pred = ( (X - Xmed)/Xsig ) @ Beta * Ysig + Ymed
        
        # Normalize
        # Handle division by zero in Xsig if any const selection
        x_sig_safe = x_sig.copy()
        x_sig_safe[x_sig_safe == 0] = 1.0
        
        x_norm = (x_sel - x_med) / x_sig_safe
        
        y_pred_norm = x_norm @ beta
        y_pred = y_pred_norm * y_sig + y_med
        
        # F. Save Predictions
        out_name = os.path.splitext(os.path.basename(fpath))[0] + "_pred.csv"
        out_path = os.path.join("results_inference", out_name)
        os.makedirs("results_inference", exist_ok=True)
        
        # Save simple CSV: Time/Index (Row), Analytes (Cols)
        header_str = "Index, CB_Pred, GL_Pred, XY_Pred" # Adjust based on 'cname' from training
        # We don't have time index here easily, use running index
        indices = np.arange(y_pred.shape[0]).reshape(-1, 1)
        output_data = np.hstack([indices, y_pred])
        
        np.savetxt(out_path, output_data, delimiter=",", header=header_str, comments='')
        print(f"Saved predictions to {out_path}")
        
    print("\nInference Complete.")

if __name__ == "__main__":
    main_inference()
