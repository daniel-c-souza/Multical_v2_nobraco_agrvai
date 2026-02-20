import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist, t as t_dist
from src.multical.core.engine import MulticalEngine
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from src.multical.models.pcr import pcr_model
from src.multical.models.spa import spa_model, spa_clean

# Extended Engine
class ExtendedMulticalEngine(MulticalEngine):
    def __init__(self):
        super().__init__()
        self.Ypred_cv_store = None
        self.X_measured_store = None
        self.best_k_ftest_store = None
        
    def run_cv(self, Selecao, x, absor, kmax, OptimModel, nc, cname, frac_test, output_dir=None, outlier=0, use_ftest=True, colors=None):
        """
        Runs Cross-Validation logic. Override to capture Ypred_cv.
        """
        # Capture measured X (unnormalized) for later use if needed, 
        # though x passed here is usually the same as x0 passed to run.
        self.X_measured_store = x

        if colors is None:
             colors = ['blue'] * nc
        
        cname = [c.strip() for c in cname]
        
        nd, nl = absor.shape
        
        # Normalization
        xmax = np.max(x, axis=0)
        xmax[xmax == 0] = 1 
        x_norm = x / xmax
        
        x_cal = x_norm
        
        # Initialize Metrics Storage
        RMSECV = np.zeros((kmax, nc))
        RMSECV_conc = np.zeros((kmax, nc))
        RMSEcal = np.zeros((kmax, nc))
        RMSEcal_conc = np.zeros((kmax, nc))
        RMSEtest = np.zeros((kmax, nc))
        RMSEtest_conc = np.zeros((kmax, nc))
        
        Ypred_cv = np.zeros((nd, nc, kmax))
        
        # Select Model Class
        model_instance = None
        if Selecao == 1:
            model_instance = PLS()

        print(f"Running CV with kmax={kmax}...")
        
        # CV/Validation Setup
        mode = 'kfold'
        val_param = 4
        cv_type = 'random'
        
        if isinstance(OptimModel, list):
             mode = OptimModel[0]
             val_param = OptimModel[1]
             if len(OptimModel) > 2:
                 cv_type = OptimModel[2]
        elif isinstance(OptimModel, int):
             if OptimModel == -1:
                 val_param = nd
             else:
                 val_param = OptimModel
        
        is_holdout = False
        val_idx_holdout = None
        train_idx_holdout = None
        
        if mode == 'Val':
            is_holdout = True
            frac_val = val_param
            if frac_val <= 0:
                 raise ValueError(f"frac_val <= 0 (Got: {frac_val})")
            
            # Random Split
            rng_indices = np.random.permutation(nd)
            n_val = int(np.floor(nd * frac_val))
            val_idx_holdout = rng_indices[:n_val]
            train_idx_holdout = rng_indices[n_val:]
            
        else:
            # k-fold setup
            folds = val_param
            
            # Determine Indices based on cv_type
            if cv_type == 'random':
                indices = np.random.permutation(nd)
            else:
                indices = np.arange(nd)
                
            fold_size = int(np.ceil(nd / folds))

        # --- OPTIMIZED FAST CV FOR PLS (Selecao == 1) ---
        if Selecao == 1:
            print("  -> Using Optimized Fast-CV (Incremental NIPALS)")
            
            # 1. Fast Calibration Error (Full Data)
            # Normalize Combined (Used x_cal, is x_norm)
            Y_cal_in, Ymed_cal, Ysig_cal = zscore_matlab_style(x_cal)
            X_cal_in, Xmed_cal, Xsig_cal = zscore_matlab_style(absor)
            
            # Run NIPALS Once
            _, _, P_all, _, Q_all, W_all, _, _ = model_instance.nipals(X_cal_in, Y_cal_in, kmax)
            
            # Incremental Prediction (Calibration)
            for k in range(1, kmax + 1):
                wk = W_all[:, :k]
                pk = P_all[:, :k]
                qk = Q_all[:, :k]
                pw = pk.T @ wk
                pw_inv = np.linalg.pinv(pw)
                Beta_k = wk @ pw_inv @ qk.T
                
                Y_cal_pred_norm = X_cal_in @ Beta_k
                Y_cal_pred = Y_cal_pred_norm * Ysig_cal + Ymed_cal
                
                diff_cal = Y_cal_pred - x_cal
                RMSEcal[k-1, :] = np.sqrt(np.mean(diff_cal**2, axis=0))

            # 2. Fast Cross-Validation
            if is_holdout:
                 indices_loop = [0] # 1 iteration
            else:
                 indices_loop = range(folds)
                 
            for i in indices_loop:
                if is_holdout:
                    train_idx = train_idx_holdout
                    val_idx = val_idx_holdout
                else:
                    # Determine Fold Indices
                    if cv_type == 'venetian':
                        val_idx = np.arange(i, nd, folds)
                    else:
                        start = i * fold_size
                        end = min((i + 1) * fold_size, nd)
                        val_idx_raw = np.arange(start, end)
                        val_idx_raw = val_idx_raw[val_idx_raw < nd]
                        val_idx = indices[val_idx_raw]
                    
                    mask = np.ones(nd, dtype=bool)
                    mask[val_idx] = False
                    train_idx = np.arange(nd)[mask]
                
                if len(val_idx) == 0: continue
                
                # Split Data
                X_train_raw = absor[train_idx, :]
                X_val_raw = absor[val_idx, :]
                Y_train_raw = x_cal[train_idx, :] # x_cal is x_norm
                
                # Normalize (Switch=1 Logic: Combined [Train; Val])
                Combined_X = np.vstack([X_train_raw, X_val_raw])
                Combined_X_norm, Xmed_cv, Xsig_cv = zscore_matlab_style(Combined_X)
                n_tr = X_train_raw.shape[0]
                X_train = Combined_X_norm[:n_tr, :]
                X_val = Combined_X_norm[n_tr:, :]
                
                # Normalize Y (Train Only)
                Y_train, Ymed_cv, Ysig_cv = zscore_matlab_style(Y_train_raw)
                
                # Run NIPALS ONCE on Train
                _, _, P_fold, _, Q_fold, W_fold, _, _ = model_instance.nipals(X_train, Y_train, kmax)
                
                # Incremental Prediction
                for k in range(1, kmax + 1):
                    wk = W_fold[:, :k]
                    pk = P_fold[:, :k]
                    qk = Q_fold[:, :k]
                    pw = pk.T @ wk
                    pw_inv = np.linalg.pinv(pw)
                    Beta_k = wk @ pw_inv @ qk.T
                    
                    Ytp_norm = X_val @ Beta_k
                    Ytp = Ytp_norm * Ysig_cv + Ymed_cv
                    
                    # Store Predictions (nd, nc, kmax)
                    Ypred_cv[val_idx, :, k-1] = Ytp

            # Calculate RMSECV from aggregated predictions
            for k_idx in range(kmax):
                 if is_holdout:
                      diff = Ypred_cv[val_idx_holdout, :, k_idx] - x_cal[val_idx_holdout]
                 else:
                      diff = Ypred_cv[:, :, k_idx] - x_cal
                 rmsecv_k = np.sqrt(np.mean(diff**2, axis=0))
                 RMSECV[k_idx, :] = rmsecv_k
                 
                 # Convert to Concentration Units
                 RMSECV_conc[k_idx, :] = RMSECV[k_idx, :] * xmax
                 RMSEcal_conc[k_idx, :] = RMSEcal[k_idx, :] * xmax

        # Output Text Files 
        k_col = np.arange(1, kmax + 1).reshape(-1, 1)

        # Helper to save with header format: k  comp1  comp2 ...
        header_str = "k\t" + "\t".join(cname)
        
        if output_dir:
            # Save RMSEcalconc (Original Units)
            data_cal = np.hstack([k_col, RMSEcal_conc])
            np.savetxt(os.path.join(output_dir, 'Erro_cal.txt'), data_cal, header=header_str, fmt='%g', delimiter='\t')
            
            # Save RMSECVconc (Original Units)
            data_cv = np.hstack([k_col, RMSECV_conc])
            np.savetxt(os.path.join(output_dir, 'Erro_cv.txt'), data_cv, header=header_str, fmt='%g', delimiter='\t')
            
            # Save RMSEcal (Normalized)
            data_cal_norm = np.hstack([k_col, RMSEcal])
            np.savetxt(os.path.join(output_dir, 'Erro_cal_norm.txt'), data_cal_norm, header=header_str, fmt='%g', delimiter='\t')
            
            # Save RMSECV (Normalized)
            data_cv_norm = np.hstack([k_col, RMSECV])
            np.savetxt(os.path.join(output_dir, 'Erro_cv_norm.txt'), data_cv_norm, header=header_str, fmt='%g', delimiter='\t')
            
            # Save minimos.txt
            min_rmse = np.min(RMSECV_conc, axis=0)
            min_idx = np.argmin(RMSECV_conc, axis=0) + 1 
            minimos = np.vstack([min_rmse, min_idx])
            np.savetxt(os.path.join(output_dir, 'minimos.txt'), minimos, fmt='%g', delimiter='\t')

        # --- Pre-calculate Model Selection (Osten F-test) ---
        best_k_ftest = {} # Dictionary to store best k for each component if f-test used
        f_test_results = {} # Store results for plotting later

        if use_ftest:
            print(f"\n--- Model Selection (Osten F-test - Dissertation Method) ---")
            for j in range(nc):
                best_k_idx_rmse = np.argmin(RMSECV_conc[:, j])
                
                f_values = []
                f_crits = []
                k_steps = []
                
                # Loop for k vs k+1
                for k_chk in range(kmax - 1):
                    k_val = k_chk + 1
                    k_steps.append(k_val) 
                    
                    rmse_sq_k = RMSECV_conc[k_chk, j]**2
                    rmse_sq_k_plus_1 = RMSECV_conc[k_chk + 1, j]**2
                    
                    if rmse_sq_k_plus_1 == 0: eps = 1e-10
                    else: eps = 0
                    
                    if is_holdout and train_idx_holdout is not None:
                        n_cal_eff = len(train_idx_holdout)
                    else:
                        n_cal_eff = nd

                    df2 = n_cal_eff - k_val - 1
                    if df2 <= 0: df2 = 1 
                    
                    numerator = rmse_sq_k - rmse_sq_k_plus_1
                    if numerator < 0:
                        F_stat = 0
                    else:
                        F_stat = (numerator / (rmse_sq_k_plus_1 + eps)) * df2
                    
                    f_values.append(F_stat)
                    f_crit_val = f_dist.ppf(0.95, 1, df2)
                    f_crits.append(f_crit_val)

                # Selection Calculation
                best_k = 1 
                reason = "Base Model (No significant improvement found)"
                
                # Check each step. If significant, update best_k to the target of the step.
                for i_k, freq_k in enumerate(k_steps):
                    # freq_k is the base k (e.g. 1). Target is freq_k + 1 (e.g. 2)
                    if f_values[i_k] >= f_crits[i_k]:
                         # Significant Improvement
                         current_target_k = freq_k + 1
                         if current_target_k > best_k:
                             best_k = current_target_k
                             reason = f"Significant Improvement at k={freq_k}->{current_target_k}"

                best_k_ftest[j] = best_k
                f_test_results[j] = {
                    'f_values': f_values,
                    'f_crits': f_crits,
                    'k_steps': k_steps,
                    'best_k': best_k,
                    'reason': reason
                }

                print(f"Component {j+1} ({cname[j]}): F-test suggested k={best_k} (Global Min k={best_k_idx_rmse+1})")
                
                # Debug print removed for brevity in plot version, or kept if needed.
                # Keeping minimal logging.


            # --- Plotting CV Results (RMSE Only) ---
            from matplotlib.ticker import MaxNLocator

            # 1. RMSECV vs Latent Variables
            fig_rmse, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
            axes = axes.flatten()
            
            for j in range(nc):
                ax = axes[j]
                c_curr = colors[j] if j < len(colors) else 'blue'
                
                # Use analyte color for RMSECV, black/gray for RMSEC to avoid clash
                ax.plot(k_col, RMSECV[:, j], color=c_curr, linestyle='-', label='RMSECV (norm)')
                ax.plot(k_col, RMSEcal[:, j], color=c_curr, linestyle='--', label='RMSEC (norm)')
                
                # Determine which k to highlight
                if use_ftest and j in best_k_ftest:
                    best_k = best_k_ftest[j]
                    best_k_idx = best_k - 1
                    label_k = f'F-test Selected (k={best_k})'
                    marker_k = 'k*'
                else:
                    best_k_idx = np.argmin(RMSECV[:, j])
                    best_k = best_k_idx + 1
                    label_k = f'Global Min (k={best_k})'
                    marker_k = 'k*' 
                
                min_r = RMSECV[best_k_idx, j]
                ax.plot(best_k, min_r, 'k*', markersize=15, label=label_k)
                
                ax.set_xlabel('Latent Variables (k)')
                ax.set_ylabel(f'RMSE (Normalized) - {cname[j] if cname else ""}')
                ax.set_title(f'RMSE Calibration/CV (Normalized) - {cname[j]}')
                ax.legend()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                # --- NEW INDIVIDUAL PLOT ---
                fig_single, ax_single = plt.subplots(figsize=(6, 5))
                ax_single.plot(k_col, RMSECV[:, j], color=c_curr, linestyle='-', label='RMSECV (norm)')
                ax_single.plot(k_col, RMSEcal[:, j], color=c_curr, linestyle='--', label='RMSEC (norm)')
                ax_single.plot(best_k, min_r, 'k*', markersize=15, label=label_k)
                
                ax_single.set_xlabel('Latent Variables (k)')
                ax_single.set_ylabel(f'RMSE (Normalized) - {cname[j] if cname else ""}')
                ax_single.set_title(f'RMSE Calibration/CV (Normalized) - {cname[j]}')
                ax_single.legend()
                ax_single.xaxis.set_major_locator(MaxNLocator(integer=True))

                if output_dir:
                    out_name = f'RMSE_CV_{cname[j]}.png'
                    fig_single.savefig(os.path.join(output_dir, out_name), dpi=300)
                plt.close(fig_single)

                
            if output_dir:
                fig_rmse.savefig(os.path.join(output_dir, 'RMSE_Calibration_CV.png'), dpi=300)
                
            plt.tight_layout()
            plt.show(block=False)
            
            self.best_k_ftest_store = best_k_ftest


        else:
             print("Warning: Only PLS (Selecao=1) is fully supported in this extended engine for Ypred extraction.")
             # Fallback to standard loop logic if needed, but not implementing here for brevity

        self.Ypred_cv_store = Ypred_cv # STORE PREDICTIONS IN CLASS ATTRIBUTE
        
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc

def main():
    # --- CONFIGURATION (Match main_multi_ILS_new.py) ---
    results_dir = "results_publication"
    os.makedirs(results_dir, exist_ok=True)
    
    Selecao = 1
    kmax = 15
    nc = 3
    cname = ['cb', 'gl', 'xy']
    colors = ['green', 'red', 'purple']
    unid = 'g/L'
    optkini = 2
    lini = 0
    frac_test = 0.0
    dadosteste = []
    
    validation_mode = 'kfold' 
    kpart = 5
    cv_type = 'venetian'
    
    if validation_mode == 'kfold':
        OptimModel = ['kfold', kpart, cv_type]
    elif validation_mode == 'Val':
        OptimModel = ['Val', 0.20]
    else:
         OptimModel = ['kfold', 5, 'venetian']

    outlier = 0
    use_ftest = True
    analysis_list = [['LB'], ['PCA']]

    data_files = [
        ('exp4_refe.txt', 'exp4_nonda.txt'),
        ('exp5_refe.txt', 'exp5_nonda.txt'),
        ('exp6_refe.txt', 'exp6_nonda.txt'),   
        ('exp7_refe.txt', 'exp7_nonda.txt'),
    ]
    
    pretreat = [
        ['Cut', 4400, 7500, 1],
        ['SG', 7, 1, 2, 1, 1],
    ]

    # --- DATA LOADING ---
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
                
                x_list.append(xi)
                absor_list.append(data_curr)
                if ti_conc is not None:
                     time_list_conc.append(ti_conc)
                time_list_spec.append(ti_spec)
                
            except Exception as e:
                print(f"Error loading {x_f}/{abs_f}: {e}")

    if x_list:
        x0 = np.vstack(x_list)
        absor_data = np.vstack(absor_list)
        absor0 = np.vstack([wavelengths, absor_data])
        print(f"Total samples loaded: {x0.shape[0]}")
    else:
        print("No valid data files loaded. Exiting.")
        return

    # --- EXECUTION ---
    engine = ExtendedMulticalEngine()
    RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = engine.run(
        Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
        frac_test, dadosteste, OptimModel, pretreat, 
        analysis_list=analysis_list, output_dir=results_dir, outlier=outlier, use_ftest=use_ftest,
        colors=colors
    )
    
    # --- PLOTTING ---
    Ypred_cv = engine.Ypred_cv_store
    x_measured = engine.X_measured_store
    
    if Ypred_cv is None:
        print("Error: Ypred_cv was not captured. Check execute logic.")
        return

    xmax = np.max(x_measured, axis=0)
    xmax[xmax == 0] = 1 
    
    # --- INDIVIDUAL PLOTS (Restored) ---
    print("Generating Individual Plots...")
    for j in range(nc):
        analyte_name = cname[j]
        
        # Determine best k
        best_k_idx = 0 # Default
        if use_ftest and hasattr(engine, 'best_k_ftest_store') and engine.best_k_ftest_store is not None:
             # Logic to retrieve best k from store if available
             # Since 'best_k_ftest_store' might be a dict or list
             if isinstance(engine.best_k_ftest_store, dict) and j in engine.best_k_ftest_store:
                 best_k = engine.best_k_ftest_store[j]
             elif isinstance(engine.best_k_ftest_store, list) and j < len(engine.best_k_ftest_store):
                 best_k = engine.best_k_ftest_store[j]
             else:
                 # Fallback to Min RMSECV
                 best_k = np.argmin(RMSECV_conc[:, j]) + 1
        else:
            best_k = np.argmin(RMSECV_conc[:, j]) + 1
            
        best_k_idx = int(best_k - 1)
        
        # Get predictions
        y_pred_norm = Ypred_cv[:, j, best_k_idx]
        y_pred_conc = y_pred_norm * xmax[j]
        y_meas_conc = x_measured[:, j]

        # Limits
        mn_val = min(y_meas_conc.min(), y_pred_conc.min())
        mx_val = max(y_meas_conc.max(), y_pred_conc.max())
        mn = mn_val - 0.05 * (mx_val - mn_val)
        mx = mx_val + 0.05 * (mx_val - mn_val)
        
        fig_ind, ax_ind = plt.subplots(figsize=(6, 5))
        ax_ind.scatter(y_meas_conc, y_pred_conc, s=80, alpha=0.7, edgecolors='k', c=colors[j], label='Measured vs Predicted')
        ax_ind.plot([mn, mx], [mn, mx], 'k--', lw=2, label='1:1 Line')
        
        # Stats
        corr = np.corrcoef(y_meas_conc, y_pred_conc)
        r2 = corr[0, 1]**2
        rmse = RMSECV_conc[best_k_idx, j]
        
        stats_text = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} {unid}\nPCs = {best_k}'
        ax_ind.text(0.05, 0.95, stats_text, transform=ax_ind.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_ind.set_xlabel(f'Measured {analyte_name} ({unid})', fontsize=12)
        ax_ind.set_ylabel(f'Predicted {analyte_name} ({unid})', fontsize=12)
        ax_ind.set_title(f'{analyte_name} Prediction', fontsize=14)
        ax_ind.grid(True, linestyle=':', alpha=0.6)
        ax_ind.legend(loc='lower right')
        
        out_path_ind = os.path.join(results_dir, f'Publication_Plot_{analyte_name}.png')
        fig_ind.savefig(out_path_ind, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)
        print(f"Saved individual plot: {out_path_ind}")

    # --- COMBINED PLOT (Gl and Xy) ---
    # Plotting Glucose (index 1) and Xylose (index 2) combined
    plot_indices = [1, 2] 
    
    # Create Combined Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()
    
    subplot_labels = ['a)', 'b)']
    
    for i, j in enumerate(plot_indices): 
        ax = axes[i]
        analyte_name = cname[j]
        c_label = subplot_labels[i]

        # Determine best k
        if use_ftest and j in engine.best_k_ftest_store:
            best_k = engine.best_k_ftest_store[j]
            selection_method = "F-test"
        else:
            best_k = np.argmin(RMSECV_conc[:, j]) + 1
            selection_method = "Global Min"
        
        best_k_idx = best_k - 1
        
        # Get predictions
        y_pred_norm = Ypred_cv[:, j, best_k_idx]
        y_pred_conc = y_pred_norm * xmax[j]
        y_meas_conc = x_measured[:, j]
        
        # Save raw data for external plotting if needed
        out_data = np.vstack([y_meas_conc, y_pred_conc]).T
        np.savetxt(os.path.join(results_dir, f'Data_Plot_{analyte_name}.txt'), out_data, header='Measured\tPredicted', fmt='%g', delimiter='\t')
        
        # Scatter
        mn_val = min(y_meas_conc.min(), y_pred_conc.min())
        mx_val = max(y_meas_conc.max(), y_pred_conc.max())
        mn = mn_val - 0.05 * (mx_val - mn_val)
        mx = mx_val + 0.05 * (mx_val - mn_val)

        # Plot points
        ax.scatter(y_meas_conc, y_pred_conc, s=60, alpha=0.8, edgecolors='k', linewidth=0.8, c=colors[j], label='Measured vs Predicted')
        
        # Ideal Line
        ax.plot([mn, mx], [mn, mx], 'k--', lw=2, label='1:1 Line')
        
        # Stats
        corr = np.corrcoef(y_meas_conc, y_pred_conc)
        r2 = corr[0, 1]**2
        rmse = RMSECV_conc[best_k_idx, j]

        # Annotations
        stats_text = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} {unid}\nPCs = {best_k}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'Measured {analyte_name} ({unid})', fontsize=12)
        ax.set_ylabel(f'Predicted {analyte_name} ({unid})', fontsize=12)
        ax.set_title(f'{c_label} {analyte_name} Prediction', fontsize=14, loc='left')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.legend(loc='lower right')
        
    plt.tight_layout()
    out_path = os.path.join(results_dir, f'Publication_Plot_Combined.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {out_path}")
    plt.close(fig)

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
