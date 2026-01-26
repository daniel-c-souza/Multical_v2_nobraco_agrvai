import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, f as f_dist
from ..models.pls import PLS

from ..models.pcr import pcr_model
from ..models.spa import spa_model, spa_clean
from ..preprocessing.pipeline import apply_pretreatment
from ..analysis import func_analysis

class MulticalEngine:
    def __init__(self):
        pass
        
    def run(self, Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, frac_test, dadosteste, OptimModel, pretreat_list, analysis_list=None, output_dir=None, outlier=0, use_ftest=True):
        """
        Main execution engine.
        """
        x0 = np.array(x0)
        absor0 = np.array(absor0)
        
        lambda0 = absor0[0, :]
        absor = absor0[1:, :]
        x = x0
        
        # Pretreatment
        print("Applying Pretreatment...")
        absor, lambda_ = apply_pretreatment(pretreat_list, absor, lambda0, output_dir=output_dir)

        # Analysis
        if analysis_list:
             print("\n--- Running Data Analysis (Post-Pretreatment) ---")
             func_analysis(analysis_list, absor, lambda_, x, block=False, output_dir=output_dir)
             print("-----------------------------\n")
        
        # Test Data Handling (dadosteste)
        # If dadosteste is provided
        xtest_final = None
        absortest_final = None
        
        if dadosteste:
            # dadosteste = (xtest, absor_test_matrix)
            xtest_in = dadosteste[0]
            absortest_in = dadosteste[1]
            if xtest_in is not None and len(xtest_in) > 0:
                lambdatest0 = absortest_in[0, :]
                absortest0 = absortest_in[1:, :]
                
                absortest_final, lambdatest_final = apply_pretreatment(pretreat_list, absortest0, lambdatest0)
                xtest_final = xtest_in

        # Consistency Check
        ndx, nlx = x.shape
        nd, nl = absor.shape
        
        if nd != ndx:
            print(f"Error: Number of samples in X ({ndx}) and Absor ({nd}) mismatch.")
            return None, None
        if nlx != nc:
             print(f"Error: Number of components ({nc}) does not match X ({nlx}).")
             return None, None
             
        # Cross Validation Setup
        # OptimModel corresponds to 'kpart' (number of folds), or -1 for LOOCV.

        
        # Run CV
        print("Running Cross Validation...")

        
        RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = self.run_cv(
            Selecao, x, absor, kmax, OptimModel, nc, cname, frac_test, 
            output_dir=output_dir, outlier=outlier, use_ftest=use_ftest
        )
        
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc


    def run_cv(self, Selecao, x, absor, kmax, OptimModel, nc, cname, frac_test, output_dir=None, outlier=0, use_ftest=True):
        """
        Runs Cross-Validation logic.
        """
        cname = [c.strip() for c in cname]
        
        nd, nl = absor.shape
        
        # Normalization
        xmax = np.max(x, axis=0)
        xmax[xmax == 0] = 1 
        x_norm = x / xmax
        
        x_cal = x_norm
        y_cal = x_norm 
        
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
        cv_type = 'random' # Default to random if not specified ('random', 'consecutive', 'venetian')
        
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

        # Loop over latent variables k
        for k_idx in range(kmax):
            k = k_idx + 1
            y_pred_k = np.zeros((nd, nc)) 
            
            if is_holdout:
                 # --- Val Mode (Hold-out) ---
                 X_train = x_cal[train_idx_holdout, :]
                 Absor_train = absor[train_idx_holdout, :]
                 Absor_val = absor[val_idx_holdout, :]
                 
                 if Selecao == 1:
                      # Predict on Validation Set
                      _, ytp_val, _ = model_instance.predict_model(Absor_train, X_train, k, Xt=Absor_val, teste_switch=1)
                      y_pred_k[val_idx_holdout, :] = ytp_val
            else:
                # --- k-Fold Cross Validation ---
                for i in range(folds):
                    if cv_type == 'venetian':
                        # Venetian Blinds: 0, k, 2k...
                        val_idx = np.arange(i, nd, folds)
                    else:
                        # Consecutive blocks (random or sorted depends on 'indices')
                        start = i * fold_size
                        end = min((i + 1) * fold_size, nd)
                        val_idx_raw = np.arange(start, end)
                        # Filter out of bounds
                        val_idx_raw = val_idx_raw[val_idx_raw < nd]
                        val_idx = indices[val_idx_raw]
                    
                    mask = np.ones(nd, dtype=bool)
                    mask[val_idx] = False
                    train_idx = np.arange(nd)[mask]
                    
                    if len(val_idx) == 0: continue
                    
                    X_train = x_cal[train_idx, :]
                    Absor_train = absor[train_idx, :]
                    Absor_val = absor[val_idx, :]
                    
                    if Selecao == 1:
                        _, ytp_fold, _ = model_instance.predict_model(Absor_train, X_train, k, Xt=Absor_val, teste_switch=1)
                        # Ensure we map predictions back to the correct rows in y_pred_k
                        # y_pred_k is (nd, nc), val_idx are the indices in the full dataset
                        y_pred_k[val_idx, :] = ytp_fold

            # Store CV Prediction for this k
            Ypred_cv[:, :, k_idx] = y_pred_k
            
            # Compute RMSECV for this k
            if is_holdout:
                 diff = y_pred_k[val_idx_holdout] - x_cal[val_idx_holdout]
            else:
                 diff = y_pred_k - x_cal
                 
            rmsecv_k = np.sqrt(np.mean(diff**2, axis=0))
            RMSECV[k_idx, :] = rmsecv_k
            
            # Calibration Error (Whole set)
            if Selecao == 1:
                 yp, _, _ = model_instance.predict_model(absor, x_cal, k, Xt=None, teste_switch=1)
                 diff_cal = yp - x_cal
                 rmsecal_k = np.sqrt(np.mean(diff_cal**2, axis=0))
                 RMSEcal[k_idx, :] = rmsecal_k

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
            
            # Save minimos.txt (based on CONC, or normalized? Original code used conc. 
            # If plots are normalized, maybe the user wants min of normalized? 
            # Usually min index is the same. Let's keep original unless requested otherwise, 
            # but I'll add minimos_norm just in case)
            min_rmse = np.min(RMSECV_conc, axis=0)
            min_idx = np.argmin(RMSECV_conc, axis=0) + 1 
            minimos = np.vstack([min_rmse, min_idx])
            np.savetxt(os.path.join(output_dir, 'minimos.txt'), minimos, fmt='%g', delimiter='\t')
        
        # --- Plotting CV Results ---
        from matplotlib.ticker import MaxNLocator

        # 1. RMSECV vs Latent Variables
        fig_rmse, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes.flatten()
        
        for j in range(nc):
            ax = axes[j]
            ax.plot(k_col, RMSECV[:, j], 'b-', label='RMSECV (norm)')
            ax.plot(k_col, RMSEcal[:, j], 'r-', label='RMSEC (norm)')
            
            # Highlight Minimum
            best_k_idx = np.argmin(RMSECV[:, j])
            best_k = best_k_idx + 1
            min_r = RMSECV[best_k_idx, j]
            ax.plot(best_k, min_r, 'g*', markersize=15, label=f'Global Min (k={best_k})')
            
            ax.set_xlabel('Latent Variables (k)')
            ax.set_ylabel(f'RMSE (Normalized) - {cname[j] if cname else ""}')
            ax.set_title(f'RMSE Calibration/CV (Normalized) - {cname[j]}')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
        if output_dir:
            fig_rmse.savefig(os.path.join(output_dir, 'RMSE_Calibration_CV.png'), dpi=300)
            
            
        plt.tight_layout()
        plt.show(block=False)

        # 2. Osten F-test Statistic Plot
        if use_ftest:
            fig_fstat, axes_f = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
            axes_f = axes_f.flatten()

        # 3. Predicted vs Measured (Best K selection via Osten F-test)
        fig_pred, axes_pred = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes_pred = axes_pred.flatten()
        
        if use_ftest:
            print(f"\n--- Model Selection (Osten F-test - Dissertation Method) ---")

        for j in range(nc):
            # --- Selection Logic & F-stats Calculation ---
            # Default k selection for plotting: Global Minimum of RMSECV
            best_k_idx_rmse = np.argmin(RMSECV_conc[:, j])
            best_k_idx_sel = best_k_idx_rmse 
            
            if use_ftest:
                f_values = []
                f_crits = []
                k_steps = [] # Records the 'k' being tested against 'k+1'
                
                best_k_idx_ftest = 0
                found_stop = False
                
                # Loop for k vs k+1
                for k_chk in range(kmax - 1):
                    k_val = k_chk + 1   # The base model k
                    k_next = k_val + 1  # The more complex model k+1
                    k_steps.append(k_val) 
                    
                    # RMSE indices assume 0-based
                    rmse_sq_k = RMSECV_conc[k_chk, j]**2
                    rmse_sq_k_plus_1 = RMSECV_conc[k_chk + 1, j]**2
                    
                    if rmse_sq_k_plus_1 == 0: eps = 1e-10
                    else: eps = 0
                    
                    # F_estat: ( (RMSE_k^2 - RMSE_{k+1}^2) / RMSE_{k+1}^2 ) * df2
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
                    
                    # F_critical: F(0.95, df1=1, df2=n_cal - k - 1)
                    f_crit_val = f_dist.ppf(0.95, 1, df2)
                    f_crits.append(f_crit_val)
                    
                    if not found_stop:
                        if F_stat < f_crit_val:
                            # Improvement NOT significant.
                            # Stop. Best model is k_val.
                            best_k_idx_ftest = k_chk
                            found_stop = True
                        else:
                            # Improvement significant. Move to next.
                            # If this is the last loop step, we end up accepting max tested.
                            best_k_idx_ftest = k_chk + 1
                
                best_k_selected_ftest = best_k_idx_ftest + 1
                print(f"Component {j+1} ({cname[j]}): F-test suggested k={best_k_selected_ftest} (Global Min k={best_k_idx_rmse+1})")
                print(f" Detailed F-test stats for {cname[j]}:")
                print(" k_base -> k_next | F_calc | F_crit | Significant? | Action")
                header_printed = True
                
                # Re-run logic for printing (or store it)
                # Since we didn't store details, we just print the stored arrays
                # k_steps has [1, 2, 3...] corresponding to 1->2, 2->3...
                current_best = 1
                stop_triggered = False
                
                for i, k_base in enumerate(k_steps):
                    f_val = f_values[i]
                    f_crit = f_crits[i]
                    is_sig = f_val >= f_crit
                    
                    action = ""
                    if not stop_triggered:
                        if is_sig:
                            current_best = k_base + 1
                            action = f"Accept k={current_best}"
                        else:
                            stop_triggered = True
                            action = f"Stop. Keep k={current_best}"
                    else:
                        action = "(Skipped by stop rule)"
                        
                    print(f" {k_base:<6} -> {k_base+1:<6} | {f_val:.4f} | {f_crit:.4f} | {str(is_sig):<12} | {action}")

                # --- Plot F-Stats ---
                ax_f = axes_f[j]
                ax_f.plot(k_steps, f_values, 'b-o', label=r'$F_{calc}$')
                ax_f.plot(k_steps, f_crits, 'r--', label=r'$F_{crit} (95\%)$')
                
                # Mark selected
                # If best_k < kmax, the stop happened at k = best_k. 
                # The test best_k vs best_k+1 failed.
                # We can highlight the point corresponding to best_k index in the check list?
                # Actually k_steps are [1, 2, ...]. 
                # If Selected k=2, it means 1->2 was significant, 2->3 was NOT.
                # So at x=1 (test 1 vs 2), F > Fcrit. At x=2 (test 2 vs 3), F < Fcrit.
                
                ax_f.set_xlabel('Latent Variables (k) vs (k+1)')
                ax_f.set_ylabel('F Statistic')
                ax_f.set_title(f'F-Test Selection - {cname[j]}')
                ax_f.legend()
                ax_f.xaxis.set_major_locator(MaxNLocator(integer=True))

            
            best_k_selected = best_k_idx_sel + 1

            # --- Plot Predicted vs Measured (for Selected K) ---
            ax = axes_pred[j]
            
            # Extract Predictions (Normalized)
            y_pred_col_norm = Ypred_cv[:, j, best_k_idx_sel]
            
            # Convert to Concentration
            y_pred_col = y_pred_col_norm * xmax[j]
            y_meas_col = x[:, j]
            
            # Filter valid validation samples
            if is_holdout:
                 if val_idx_holdout is not None:
                      indices_to_plot = val_idx_holdout
                 else:
                      indices_to_plot = []
                 label_pt = "Val Set"
            else:
                 indices_to_plot = np.arange(nd)
                 label_pt = "CV Pred"
            
            if len(indices_to_plot) > 0:
                 x_plot = y_meas_col[indices_to_plot]
                 y_plot = y_pred_col[indices_to_plot]
                 
                 # Plot Scatter
                 ax.scatter(x_plot, y_plot, alpha=0.7, label=label_pt)
                 
                 # Ideal Line (1:1)
                 min_val = min(x_plot.min(), y_plot.min())
                 max_val = max(x_plot.max(), y_plot.max())
                 ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
                 
                 # Stats
                 correlation_matrix = np.corrcoef(x_plot, y_plot)
                 r2 = correlation_matrix[0, 1]**2 if correlation_matrix.shape[0] > 1 else 0
                 rmsecv_best = RMSECV_conc[best_k_idx_sel, j]
                 
                 ax.set_xlabel(f'Measured {cname[j]}')
                 ax.set_ylabel(f'Predicted {cname[j]}')
                 ax.set_title(f'{cname[j]}: k={best_k_selected}, $R^2$={r2:.3f}, RMSE={rmsecv_best:.4g}')
                 ax.legend()
        
        if output_dir:
            fig_pred.savefig(os.path.join(output_dir, 'Predicted_vs_Measured.png'), dpi=300)
                 
        
        plt.tight_layout()
        plt.show(block=False)
            
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc
