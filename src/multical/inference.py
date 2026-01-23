import numpy as np
import os
import matplotlib.pyplot as plt
from .preprocessing.pipeline import apply_pretreatment
from .analysis import func_analysis
from .models.pls import PLS
from .models.spa import spa_model, spa_clean
from .models.pcr import pcr_model
from .utils import zscore_matlab_style

def run_inference(Selecao, optkini, lini, kinf, nc, cname, unid, x0, absor0, xinf0, absorinf0, pretreat_list, pretreatinf_list, analysis_list=None, analysisinf_list=None, leverage=0, output_dir=None, kmax_spa=20):
    """
    Inference Engine.
    Scales to units of x0.
    """
    
    # 1. Prepare Data
    lambda0 = absor0[0, :]
    absor = absor0[1:, :]
    x = x0
    
    # xinf0 can be None or empty
    lambdainf = None
    absorinf = None
    xinf = None
    
    if absorinf0 is not None and len(absorinf0) > 0:
        lambdainf = absorinf0[0, :]
        absorinf = absorinf0[1:, :]
    
    if xinf0 is not None and len(xinf0) > 0:
        xinf = xinf0
        
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    if analysis_list is None: analysis_list = []
    if analysisinf_list is None: analysisinf_list = []

    # 2. Pretreatment
    print("Applying Pretreatment (Calibration)...")
    absor, lambda_ = apply_pretreatment(pretreat_list, absor, lambda0, output_dir=output_dir, prefix="Cal_")
    
    if analysis_list:
        print("Running Analysis (Calibration)...")
        func_analysis(analysis_list, absor, lambda_, x, block=False, output_dir=output_dir, prefix="Cal_")

    print("Applying Pretreatment (Inference)...")
    # For inference, if we have data
    if absorinf is not None:
         absorinf, lambdainf_ = apply_pretreatment(pretreatinf_list, absorinf, lambdainf, output_dir=output_dir, prefix="Inf_")
         if analysisinf_list:
              print("Running Analysis (Inference)...")
              # xinf might be None, pass None in that case
              func_analysis(analysisinf_list, absorinf, lambdainf_, xinf, block=False, output_dir=output_dir, prefix="Inf_")
    else:
        print("No inference absorbance data provided.")
        return None

    # 3. Consistency Checks
    nd, nl = absor.shape
    ndx, ncx = x.shape
    ndinf, nlinf = absorinf.shape
    
    if nd != ndx:
        raise ValueError(f"Calibration: X rows ({ndx}) != Absor rows ({nd})")
    if ncx != nc:
        raise ValueError(f"Calibration: Components ({nc}) != X cols ({ncx})")
    
    if nl != nlinf:
         # In reality, pretreatment might change wavelengths (e.g. Cut). 
         # Assuming pretreatments are identical or result in same wavelengths.
         # Ideally we should interpolate or check.
         print(f"Warning: Calibration wavelengths ({nl}) != Inference wavelengths ({nlinf}). This might cause errors.")
         # If they are different but same range, maybe okay? 
         # But models usually expect same number of features.
    
    # Check kinf
    max_k_possible = min(nd, nl)
    if np.any(np.array(kinf) > max_k_possible):
         print(f"Warning: Some kinf values > max possible ({max_k_possible}). They will be capped.")
         kinf = [min(k, max_k_possible) for k in kinf]

    # 4. Normalization of X (Calibration Concentrations)
    xmax = np.max(x, axis=0)
    xmax[xmax == 0] = 1 # Avoid div by zero
    x_norm = x / xmax
    
    # 5. SPA Optimization Logic (if needed)
    cini = 0
    if Selecao == 2: # SPA
        if optkini == 2:
             print("Optimizing SPA lini (optkini=2)... This may take time.")
             # Scilab: for ind0 = 1:nl, ls0(:,ind0) = spa_clean(absor, ind0, kmax)
             # Then counts most frequent index?
             # "conta(ind0) = sum(inds)" -> How many times ind0 appeared in ls0?
             # No, "inds = find(ls0==ind0)".
             # The Scilab code counts frequency of each wavelength being selected across all runs?
             # "ls0(:,ind0) = spa_clean(absor,ind0,kmax)" -> spa_clean returns list of selected indices considering ind0 as start.
             # So we run SPA forcing each variable as start.
             # Then we pick the variable that appears most often in all these subsets?
             
             # Implementing as per Scilab logic roughly:
             from collections import Counter
             
             # We need a kmax for SPA optimization. Scilab uses 'kmax'. 
             # I added kmax_spa argument.
             
             all_selected_indices = []
             for ind0 in range(nl):
                 # spa_clean in python (from previous verification) returns 'sel_cols'
                 # Note: python is 0-indexed, scilab 1-indexed.
                 sel_cols = spa_clean(absor, ind0, kmax_spa)
                 all_selected_indices.extend(sel_cols)
             
             counts = Counter(all_selected_indices)
             # Most common
             best_idx = counts.most_common(1)[0][0]
             cini = best_idx
             lini_val = lambda_[cini]
             print(f"Optimal SPA start wavelength: {lini_val} nm (Index {cini})")
             
        elif optkini == 1:
             # lini given
             # find index clini closest to lini
             idx = (np.abs(lambda_ - lini)).argmin()
             cini = idx
             print(f"Using provided SPA start: {lambda_[cini]} nm (Index {cini})")
        else:
             cini = 0
             print(f"Using first wavelength as SPA start: {lambda_[cini]}")
             
    # 6. Inference Loop
    Xinf_pred = np.zeros((ndinf, nc))
    Xcal_pred = np.zeros((nd, nc)) # To return calibration fit as well? Scilab does not seem to output it clearly but calculates it.
    
    # Helper models
    pls_obj = PLS()
    
    print("Starting Inference per component...")
    for j in range(nc):
        k = kinf[j]
        # x_norm column j
        y_cal_j = x_norm[:, j].reshape(-1, 1)
        
        # Select Model
        # Scilab: [Xp(:,j), Xinf(:,j), Par_norm] = model(...)
        # Xp is calibration prediction, Xinf is inference prediction.
        
        yp_cal_j = None
        yp_inf_j = None
        
        if Selecao == 1: # PLS
            # predict_model(X, Y, k, Xt, teste_switch=1)
            # Returns: Yp (cal), Ytp (test), Stats
            # Note: predict_model signature in pls.py: predict_model(self, X, Y, k, Xt=None, teste_switch=1)
            # Note X is absor, Y is conc.
            
             _, yp_inf_j, _ = pls_obj.predict_model(absor, y_cal_j, k, Xt=absorinf, teste_switch=1)
             
             # To get calibration prediction (Xp)
             yp_cal_j, _, _ = pls_obj.predict_model(absor, y_cal_j, k, Xt=None, teste_switch=1)
             
        elif Selecao == 2: # SPA
             # spa_model(Xcal, Ycal, k, cini, Xval)
             # We need to import spa_model. Assuming it follows similar pattern.
             yp_cal_j, yp_inf_j, _ = spa_model(absor, y_cal_j, k, cini, absorinf)
             
        elif Selecao == 3: # PCR
             # pcr_model(Xcal, Ycal, k, Xval)
             yp_cal_j, yp_inf_j, _ = pcr_model(absor, y_cal_j, k, absorinf)
             
        # Store results
        if yp_inf_j is not None:
            Xinf_pred[:, j] = yp_inf_j.flatten()
            
        if yp_cal_j is not None:
            Xcal_pred[:, j] = yp_cal_j.flatten()
            
        # Leverage Analysis (Optional)
        if leverage == 1:
             # Calculate Hii...
             pass # Not implementing leverage for now unless requested, to keep it simple.

    # 7. Denormalize Predictions
    # Xinf_conc = Xinf.*repmat(xmax,ninf,1)
    Xinf_conc = Xinf_pred * xmax
    Xcal_conc = Xcal_pred * xmax
    
    # 8. Results & Plotting
    print("Inference Complete.")
    print("Predicted Concentrations (Head):")
    print(Xinf_conc[:5, :])
    
    if output_dir:
        np.savetxt(os.path.join(output_dir, "Xinf.txt"), Xinf_conc, fmt='%.6f', delimiter='\t')
        np.savetxt(os.path.join(output_dir, "Predicted_Inference.txt"), Xinf_conc, fmt='%.6f')
    
    if xinf is not None:
        # Plot Expected vs Predicted
        print("Plotting Expected vs Predicted...")
        for j in range(nc):
            fig = plt.figure()
            y_ref = xinf[:, j]
            y_pred = Xinf_conc[:, j]
            
            # Simple metrics
            rmsep = np.sqrt(np.mean((y_ref - y_pred)**2))
            
            plt.plot(y_ref, y_pred, 'o', label='Samples')
            
            # 1:1 line
            min_val = min(np.min(y_ref), np.min(y_pred))
            max_val = max(np.max(y_ref), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
            
            plt.title(f"Prediction: {cname[j]} (RMSEP={rmsep:.4f})")
            plt.xlabel(f"Reference ({unid})")
            plt.ylabel(f"Predicted ({unid})")
            plt.legend()
            
            plt.show(block=False)
            
            if output_dir:
                 fig.savefig(os.path.join(output_dir, f"Pred_vs_Ref_{cname[j]}.png"))

    return Xinf_conc
