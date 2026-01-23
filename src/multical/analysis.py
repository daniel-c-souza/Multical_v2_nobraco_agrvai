import numpy as np
import matplotlib.pyplot as plt
import os
from src.multical.utils import zscore_matlab_style

def func_analysis(analysis_list, absor, wavelengths, x=None, block=True, output_dir=None, prefix=""):
    """
    Python equivalent of func_analysis.sci
    
    Parameters:
    analysis_list (list): List of list/tuples, e.g. [['LB'], ['PCA']]
    absor (np.ndarray): Absorbance matrix (samples x wavelengths)
    wavelengths (np.ndarray): Wavelength vector or similar ID for x-axis
    x (np.ndarray, optional): Concentration matrix (samples x analytes) for LB
    block (bool): Whether to block execution on plots (plt.show())
    output_dir (str, optional): Directory to save plots.
    prefix (str, optional): Prefix for log messages and saved files.
    """
    
    # Ensure numpy arrays
    absor = np.array(absor)
    if wavelengths is not None:
        wavelengths = np.array(wavelengths)
    if x is not None:
        x = np.array(x)

    print(f'-- {prefix}Analysis --')
    for i, anal in enumerate(analysis_list):
        name = anal[0]
        print(f"{i+1} - {name}")
        
        if name == 'LB':
            if x is None or x.size == 0:
                print(f"WARNING: cannot run LB {prefix}- no x matrix")
                continue
            
            # --- LB Logic ---
            # Case 1: No intercept (K = x \ absor) -> Minimize || xK - absor ||
            # np.linalg.lstsq(a, b) solves ax = b
            K, _, _, _ = np.linalg.lstsq(x, absor, rcond=None)
            absorc = x @ K
            
            # Global min/max for plotting limits
            if absor.size == 0 or absorc.size == 0:
                 print(f"Warning: LB Analysis {prefix}- empty arrays. Skipping.")
                 continue

            xymax = max(np.max(absor), np.max(absorc))
            xymin = min(np.min(absor), np.min(absorc))
            
            # Prepare colors (by wavelength)
            nd, nl = absor.shape
            if wavelengths is not None and len(wavelengths) == nl:
                c_vals = np.tile(wavelengths, nd)
                c_label = 'Wavelength (nm)'
            else:
                c_vals = np.tile(np.arange(nl), nd)
                c_label = 'Wavelength Index'

            # Plot 1
            fig1, ax1 = plt.subplots()
            sc1 = ax1.scatter(absor.flatten(), absorc.flatten(), c=c_vals, cmap='viridis', alpha=0.5, label='Data')
            plt.colorbar(sc1, ax=ax1, label=c_label)
            
            # Diagonal line
            ax1.plot([xymin, xymax], [xymin, xymax], '-k', label='Identity')
            ax1.set_xlabel('Observed Absorbance')
            ax1.set_ylabel('Calculated Absorbance (LB)')
            ax1.set_title(f'{prefix}LB: Fit without independent term')
            
            # Case 2: With intercept
        
            xone = np.hstack([np.ones((nd, 1)), x])
            K_bias, _, _, _ = np.linalg.lstsq(xone, absor, rcond=None)
            absorc_bias = xone @ K_bias
            
            # Plot 2
            fig2, ax2 = plt.subplots()
            sc2 = ax2.scatter(absor.flatten(), absorc_bias.flatten(), c=c_vals, cmap='viridis', alpha=0.5, label='Data')
            plt.colorbar(sc2, ax=ax2, label=c_label)
            
            ax2.plot([xymin, xymax], [xymin, xymax], '-k', label='Identity')
            ax2.set_xlabel('Observed Absorbance')
            ax2.set_ylabel('Calculated Absorbance (LB)')
            ax2.set_title(f'{prefix}LB: Fit with independent term')
            
            if output_dir:
                safe_prefix = prefix.replace(" ", "_")
                fig1.savefig(os.path.join(output_dir, f'{safe_prefix}Analysis_LB_NoIntercept.png'))
                fig2.savefig(os.path.join(output_dir, f'{safe_prefix}Analysis_LB_WithIntercept.png'))

            
        elif name == 'PCA':
            # --- PCA Logic ---
            # Standardize (Z-score)
            anorm, _, _ = zscore_matlab_style(absor)
            
            # Covariance/Correlation matrix proxy: anorm' * anorm
            
            cov_mat = anorm.T @ anorm
            
            # Eigen decomposition
            
            eigvals, eigvecs = np.linalg.eigh(cov_mat)
            
            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Plot 1: Loadings (Eigenvectors)
            fig3, ax3 = plt.subplots()
            if wavelengths is not None and len(wavelengths) == eigvecs.shape[0]:
                x_axis = wavelengths
                xlabel_text = 'Wavelength (nm)'
            else:
                x_axis = np.arange(eigvecs.shape[0])
                xlabel_text = 'Index'

            ax3.plot(x_axis, eigvecs[:, 0], label='PC1')
            ax3.plot(x_axis, eigvecs[:, 1], label='PC2')
            ax3.plot(x_axis, eigvecs[:, 2], label='PC3')
            ax3.set_xlabel(xlabel_text)
            ax3.set_ylabel('PC Loading')
            ax3.set_title('Principal Components (Loadings)')
            ax3.legend()
            ax3.plot([np.min(x_axis), np.max(x_axis)], [0, 0], 'k--')

            # Plot 2: Scores (Projection)
            
            scores = absor @ eigvecs
            fig4, ax4 = plt.subplots()
            ax4.scatter(scores[:, 0], scores[:, 1], marker='x')
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC2')
            ax4.set_title('Score Plot (PC1 vs PC2)')
            
            if output_dir:
                fig3.savefig(os.path.join(output_dir, 'Analysis_PCA_Loadings.png'))
                fig4.savefig(os.path.join(output_dir, 'Analysis_PCA_Scores.png'))
            
            # Variance usage
            vartot = np.sum(eigvals)
            var_rel = eigvals / vartot
            var_ac = np.cumsum(var_rel)
            
            # Display explained variance
            ind = np.where(var_ac < 0.9999)[0]
            maxind = max(np.max(ind) if ind.size > 0 else 0, 10)
            maxind = min(maxind, len(var_rel))
            
            print('Explained variance first PCs')
            print(f"{'PC#':<5} {'var_relative':<15} {'var_accumulated':<15}")
            for j in range(maxind):
                print(f"{j+1:<5d} {var_rel[j]:<15.5f} {var_ac[j]:<15.5f}")

    if block:
        print("Pausing for plots... (Close windows to continue)")
        plt.show()
