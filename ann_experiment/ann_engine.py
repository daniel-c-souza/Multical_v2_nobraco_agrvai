import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, f as f_dist
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style

from src.multical.models.pcr import pcr_model
from src.multical.models.spa import spa_model, spa_clean
from src.multical.models.svr import SVRModel
from src.multical.models.ann_torch import ANNModel
from src.multical.preprocessing.pipeline import apply_pretreatment
from src.multical.analysis import func_analysis

class MulticalEngine:
    def __init__(self):
        pass
        
    def run(self, Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, frac_test, dadosteste, OptimModel, pretreat_list, analysis_list=None, output_dir=None, outlier=0, use_ftest=True, colors=None):
        """
        Main execution engine.
        """
        if colors is None:
             # Default loop of colors if not provided
             colors = ['blue'] * nc
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
        print("Running Cross Validation...")

        RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc = self.run_cv(
            Selecao, x, absor, kmax, OptimModel, nc, cname, frac_test, 
            output_dir=output_dir, outlier=outlier, use_ftest=use_ftest, colors=colors
        )
        
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc


    def run_cv(self, Selecao, x, absor, kmax, OptimModel, nc, cname, frac_test, output_dir=None, outlier=0, use_ftest=True, colors=None):
        """
        Runs Cross-Validation logic.
        """
        if colors is None:
             colors = ['blue'] * nc
        
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
            
            rng_indices = np.random.permutation(nd)
            n_val = int(np.floor(nd * frac_val))
            val_idx_holdout = rng_indices[:n_val]
            train_idx_holdout = rng_indices[n_val:]
            
        else:
            folds = val_param
            if cv_type == 'random':
                indices = np.random.permutation(nd)
            else:
                indices = np.arange(nd)
            fold_size = int(np.ceil(nd / folds))

        # --- ANN Implementation (Selecao == 5) ---
        if Selecao == 5:
            print("  -> Using ANN Cross-Validation (Full Spectrum + Grid Search)")
            model_instance = ANNModel()
            
            # --- Define Grid Search Space (Huge Run ~1000 configs) ---
            # 1. Define Ranges
            # Hidden Neurons: [5, 10, 25, 50, 75, 100, 200]
            grid_hidden = [5, 10, 25, 50, 75, 100, 200]
            
            # Layers: Deep networks up to 8 layers
            grid_layers = [1, 2, 3, 4, 5, 6, 7, 8]
            
            # Activation: At least 6 functions
            grid_act = ['relu', 'tanh', 'sigmoid', 'silu', 'leaky_relu', 'elu']
            
            # Learning Rate: Standard Log Space
            grid_lr = [0.01, 0.001, 0.0001]
            
            epochs_fixed = 500
            # Disable Early Stopping to prioritize accuracy for the overnight run
            early_stop = False
            patience_val = 20
            
            # 2. Flatten Grid
            param_list = []
            for h in grid_hidden:
                for lay in grid_layers:
                    for act in grid_act:
                        for lr in grid_lr:
                            param_list.append({'hidden': h, 'layers': lay, 'act': act, 'lr': lr})
            
            total_params = len(param_list)
            if kmax > total_params:
                print(f"     Note: kmax ({kmax}) > Total Combinations ({total_params}). Limiting to {total_params}.")
                kmax = total_params
            elif kmax < total_params:
                print(f"     Note: kmax ({kmax}) < Total Combinations ({total_params}). Truncating grid.")
                param_list = param_list[:kmax]
                
            print(f"     Running {len(param_list)} ANN combinations (Epochs={epochs_fixed}, EarlyStop={early_stop})...")
            print(f"     Architecture: Single-Output Models (Training one model per component per grid setup)")

            for k_idx in tqdm(range(kmax), desc="ANN Grid Search", unit="config"):
                params = param_list[k_idx]
                h = params['hidden']
                lay = params['layers']
                act = params['act']
                lr = params['lr']
                
                # Iterate over each component separately
                for j in range(nc):
                    y_target_col = x_cal[:, j:j+1] # Single column [N, 1]

                    # 1. Calibration (Full Data)
                    # Training on full calibration data (Single Output)
                    y_cal_pred = model_instance.fit_predict(absor, y_target_col, absor, 
                                                            hidden_units=h, n_layers=lay, activation=act,
                                                            learning_rate=lr, epochs=epochs_fixed,
                                                            early_stopping=early_stop, patience=patience_val)
                    
                    # y_cal_pred is [N, 1]. diff is [N, 1]
                    diff_cal = y_cal_pred - y_target_col
                    RMSEcal[k_idx, j] = np.sqrt(np.mean(diff_cal**2)) 
                    
                    # 2. CV
                    if is_holdout:
                         indices_loop = [0]
                    else:
                         indices_loop = range(folds)
                         
                    y_pred_cv_accum_col = np.zeros((nd, 1))

                    for i in indices_loop:
                        if is_holdout:
                            train_idx = train_idx_holdout
                            val_idx = val_idx_holdout
                        else:
                            mask = np.ones(nd, dtype=bool)
                            if cv_type == 'venetian':
                                val_idx = np.arange(i, nd, folds)
                            else:
                                start = i * fold_size
                                end = min((i + 1) * fold_size, nd)
                                val_idx_raw = np.arange(start, end)
                                val_idx_raw = val_idx_raw[val_idx_raw < nd]
                                val_idx = indices[val_idx_raw]
                            
                            mask[val_idx] = False
                            train_idx = np.arange(nd)[mask]
                        
                        if len(val_idx) == 0: continue

                        X_train = absor[train_idx, :]
                        Y_train = y_target_col[train_idx, :] # Single column
                        X_val = absor[val_idx, :]

                        y_val_pred = model_instance.fit_predict(X_train, Y_train, X_val, 
                                                                hidden_units=h, n_layers=lay, activation=act,
                                                                learning_rate=lr, epochs=epochs_fixed,
                                                                early_stopping=early_stop, patience=patience_val)
                        y_pred_cv_accum_col[val_idx, :] = y_val_pred
                    
                    # Store accumulated CV predictions for this component and this k_idx
                    Ypred_cv[:, j, k_idx] = y_pred_cv_accum_col.flatten()
                    
                    # Calculate RMSECV for this component
                    if is_holdout:
                         diff = y_pred_cv_accum_col[val_idx_holdout] - y_target_col[val_idx_holdout]
                    else:
                         diff = y_pred_cv_accum_col - y_target_col
                         
                    RMSECV[k_idx, j] = np.sqrt(np.mean(diff**2))
                    
                # Concatenation and storage for conc units
                RMSECV_conc[k_idx, :] = RMSECV[k_idx, :] * xmax
                RMSEcal_conc[k_idx, :] = RMSEcal[k_idx, :] * xmax

            # Report Best Params
            print("\n  --- ANN Best Parameters ---")
            for j in range(nc):
                # Search only up to kmax (which is what we actually ran)
                # RMSECV_conc might be larger (initialized to input kmax 200) than actual kmax used (20)
                # So we slice it to [:kmax]
                errors_run = RMSECV_conc[:kmax, j]
                best_idx = np.argmin(errors_run)
                
                # Check bounds
                if best_idx < len(param_list):
                    best_p = param_list[best_idx]
                    print(f"  Component {cname[j]}: Best Idx={best_idx+1}, RMSE={errors_run[best_idx]:.4f}, Hidden={best_p['hidden']}, Layers={best_p['layers']}, Act={best_p['act']}")
                else:
                    print(f"  Component {cname[j]}: Best Idx={best_idx+1} (Out of Param List Bounds)")

        else:
            # Fallback for PLS/PCR if needed (though not expected in this specific engine file)
            print("Mode not supported in ANN Engine.")
                
        # Output Text Files 
        # Resize arrays if kmax was adjusted (e.g. in ANN Grid Search)
        RMSEcal = RMSEcal[:kmax, :]
        RMSECV = RMSECV[:kmax, :]
        RMSEcal_conc = RMSEcal_conc[:kmax, :]
        RMSECV_conc = RMSECV_conc[:kmax, :]

        k_col = np.arange(1, kmax + 1).reshape(-1, 1)
        header_str = "k\t" + "\t".join(cname)
        
        if output_dir:
            data_cal = np.hstack([k_col, RMSEcal_conc])
            np.savetxt(os.path.join(output_dir, 'Erro_cal.txt'), data_cal, header=header_str, fmt='%g', delimiter='\t')
            data_cv = np.hstack([k_col, RMSECV_conc])
            np.savetxt(os.path.join(output_dir, 'Erro_cv.txt'), data_cv, header=header_str, fmt='%g', delimiter='\t')
            data_cal_norm = np.hstack([k_col, RMSEcal])
            np.savetxt(os.path.join(output_dir, 'Erro_cal_norm.txt'), data_cal_norm, header=header_str, fmt='%g', delimiter='\t')
            data_cv_norm = np.hstack([k_col, RMSECV])
            np.savetxt(os.path.join(output_dir, 'Erro_cv_norm.txt'), data_cv_norm, header=header_str, fmt='%g', delimiter='\t')
            
            min_rmse = np.min(RMSECV_conc, axis=0)
            min_idx = np.argmin(RMSECV_conc, axis=0) + 1 
            minimos = np.vstack([min_rmse, min_idx])
            np.savetxt(os.path.join(output_dir, 'minimos.txt'), minimos, fmt='%g', delimiter='\t')
        
        # --- Plotting CV Results ---
        fig_rmse, axes = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes = axes.flatten()
        
        for j in range(nc):
            ax = axes[j]
            c_curr = colors[j] if j < len(colors) else 'blue'
            
            ax.plot(k_col, RMSECV[:, j], color=c_curr, linestyle='-', label='RMSECV (norm)')
            ax.plot(k_col, RMSEcal[:, j], color=c_curr, linestyle='--', label='RMSEC (norm)')
            
            best_k_idx = np.argmin(RMSECV[:, j])
            best_k = best_k_idx + 1
            label_k = f'Min (Idx={best_k})'
            
            min_r = RMSECV[best_k_idx, j]
            ax.plot(best_k, min_r, 'k*', markersize=15, label=label_k)
            
            ax.set_xlabel('Grid Index')
            ax.set_ylabel(f'RMSE (Normalized)')
            ax.set_title(f'{cname[j]}')
            ax.legend()
            
        if output_dir:
            fig_rmse.savefig(os.path.join(output_dir, 'RMSE_Calibration_CV.png'), dpi=300)
            
        plt.tight_layout()
        # plt.show(block=False)

        # Predicted vs Measured
        fig_pred, axes_pred = plt.subplots(1, nc, figsize=(6*nc, 5), squeeze=False)
        axes_pred = axes_pred.flatten()

        for j in range(nc):
            best_k_idx_sel = np.argmin(RMSECV_conc[:, j])
            
            ax = axes_pred[j]
            y_pred_col_norm = Ypred_cv[:, j, best_k_idx_sel]
            y_pred_col = y_pred_col_norm * xmax[j]
            y_meas_col = x[:, j]
            
            if is_holdout:
                 indices_to_plot = val_idx_holdout if val_idx_holdout is not None else []
                 label_pt = "Val Set"
            else:
                 indices_to_plot = np.arange(nd)
                 label_pt = "CV Pred"
            
            if len(indices_to_plot) > 0:
                 x_plot = y_meas_col[indices_to_plot]
                 y_plot = y_pred_col[indices_to_plot]
                 ax.scatter(x_plot, y_plot, alpha=0.7, label=label_pt, color=colors[j])
                 
                 min_val = min(x_plot.min(), y_plot.min())
                 max_val = max(x_plot.max(), y_plot.max())
                 ax.plot([min_val, max_val], [min_val, max_val], 'k--')
                 
                 ax.set_xlabel(f'Measured {cname[j]}')
                 ax.set_ylabel(f'Predicted {cname[j]}')
                 ax.legend()
        
        if output_dir:
            fig_pred.savefig(os.path.join(output_dir, 'Predicted_vs_Measured.png'), dpi=300)
                 
        plt.tight_layout()
        # plt.show(block=False)
            
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc, RMSEtest, RMSEtest_conc
