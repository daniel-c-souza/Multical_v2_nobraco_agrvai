import numpy as np
import matplotlib.pyplot as plt
import os
from src.multical.utils import zscore_matlab_style

def calculate_vip(model, X, Y, k, wavelengths=None, output_dir=None):
    """
    Calculates Variable Importance in Projection (VIP) scores.
    Formula: VIP_j = sqrt(p * sum(SSY_a * (w_ja / ||w_a||)^2) / sum(SSY_a))
    """
    # Normalize Inputs to fit model manually
    Xnorm, Xmed, Xsig = zscore_matlab_style(X)
    Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
    
    # Fit PLS
    B, T, P, U_scores, Q_loadings, W, r2X, r2Y = model.nipals(Xnorm, Ynorm, k)
    
    # Calculate VIP
    p = X.shape[1] # Number of variables
    h = k          # Number of components used
    
    vip_scores = np.zeros(p)
    ssy_total = np.sum(r2Y[:h]) # Total explained variance (since r2Y is percentage)
    
    # W is matrix of weights (p x h)
    # r2Y is explained variance per component
    
    for j in range(p):
        weight_sum = 0.0
        for a in range(h):
            # w_ja is W[j, a]
            # NIPALS W are normalized? pls.py says "w = w / np.linalg.norm(w)". So ||w_a|| = 1.
            # So term is simply w_ja^2
            weight_sum += r2Y[a] * (W[j, a]**2)
            
        vip_scores[j] = np.sqrt(p * weight_sum / ssy_total)

    if output_dir:
        # Plot VIP
        fig, ax = plt.subplots(figsize=(10, 5))
        if wavelengths is not None:
            ax.plot(wavelengths, vip_scores)
            ax.set_xlabel("Wavelength")
            # Mark VIP=1 threshold
            ax.axhline(y=1.0, color='r', linestyle='--', label='VIP > 1')
        else:
            ax.plot(vip_scores)
            ax.set_xlabel("Variable Index")
            ax.axhline(y=1.0, color='r', linestyle='--')
            
        ax.set_ylabel("VIP Score")
        ax.set_title(f"Variable Importance in Projection (VIP) - {h} LVs")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "VIP_Scores.png"))
        plt.show(block=False)
        
    return vip_scores
