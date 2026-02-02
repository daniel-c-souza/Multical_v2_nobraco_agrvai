import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.multical.models.svr import SVRModel
from src.multical.preprocessing.pipeline import apply_pretreatment

class MulticalEngineSVR:
    def __init__(self):
        pass
        
    def run(self, Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, frac_test, dadosteste, OptimModel, pretreat_list, analysis_list=None, output_dir=None, outlier=0, use_ftest=True, colors=None):
        if colors is None: colors = ['blue'] * nc
        x0 = np.array(x0)
        absor0 = np.array(absor0)
        
        lambda0 = absor0[0, :]
        absor = absor0[1:, :]
        x = x0 # x corresponds to the reference values (concentrations)
        
        # Pretreatment
        print("Applying Pretreatment...")
        absor, lambda_ = apply_pretreatment(pretreat_list, absor, lambda0, output_dir=output_dir)

        ndx, nlx = x.shape
        nd, nl = absor.shape
        
        # Cross Validation Setup
        print("Running Cross Validation (SVR)...")
        RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc = self.run_cv(
            x, absor, kmax, OptimModel, nc, cname, output_dir, colors
        )
        
        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc

    def run_cv(self, x, absor, kmax, OptimModel, nc, cname, output_dir=None, colors=None):
        if colors is None: colors = ['blue'] * nc
        cname = [c.strip() for c in cname]
        nd, nl = absor.shape
        
        # Normalization of Y (Concentrations)
        # This is CRITICAL for SVR to converge properly if targets have large range
        xmax = np.max(x, axis=0)
        xmax[xmax == 0] = 1 
        x_norm = x / xmax
        
        x_cal = x_norm
        
        # Grid Setup
        # Reasonable SVR Grid
        grid_C = [0.1, 1, 10, 100, 1000, 5000, 10000, 50000]
        grid_epsilon = [0.0001, 0.001, 0.01, 0.05, 0.1]
        grid_gamma = ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001]
        grid_kernel = ['rbf', 'sigmoid', 'poly', 'linear'] # Expanded kernels
        
        param_list = []
        for c in grid_C:
            for eps in grid_epsilon:
                for g in grid_gamma:
                    for k in grid_kernel:
                        # Optimization: Skip gamma iteration for linear kernel (it's unused)
                        if k == 'linear' and g != 'scale': 
                            continue # 'scale' acts as the one placeholder
                            
                        param_list.append({'C': c, 'epsilon': eps, 'gamma': g, 'kernel': k})
        
        total_params = len(param_list)
        # Allow kmax to limit the search if user wants a quick run
        if kmax > total_params: 
            kmax = total_params
        else: 
            pass # Keep kmax as is, truncating the list
            # Actually, let's just use the full list if kmax is arbitrary, 
            # or treat kmax as "run all" if it's large.
            # In the original code kmax was likely LV. Here it's iterations.
            
        # If the user passes a very large kmax, we just run all combinations
        active_params = param_list[:kmax]
        
        print(f"     Running {len(active_params)} SVR combinations...")

        # Initialize Storage
        # dimensions: [Iteration, Analyte]
        RMSECV = np.zeros((kmax, nc))
        RMSECV_conc = np.zeros((kmax, nc))
        RMSEcal = np.zeros((kmax, nc))
        RMSEcal_conc = np.zeros((kmax, nc))
        
        model_instance = SVRModel()
        
        # CV Setup
        mode = OptimModel[0]
        val_param = OptimModel[1]
        cv_type = OptimModel[2] if len(OptimModel) > 2 else 'random'
        
        folds = val_param
        if cv_type == 'random' or cv_type == 'kfold': # Handle standard naming
             indices = np.random.permutation(nd)
        else: 
             indices = np.arange(nd) # Venetian or Sequential
             
        fold_size = int(np.ceil(nd / folds))

        for k_idx in tqdm(range(kmax), desc="SVR Grid Search", unit="config"):
            params = active_params[k_idx]
            
            # Independent Grid Search per Component (Single Output assumption for tuning)
            # Actually SVRModel (Wrapper) handles multi-output internally by fitting N regressors.
            # But here we want to calculate RMSE for each component.
            
            # The Wrapper 'fit_predict' takes (X_train, Y_train, X_test)
            # Y_train shape (samples, nc)
            
            # --- 1. Calibration (Full Data) ---
            # We fit on all data to get Calibration Error
            y_cal_pred = model_instance.fit_predict(absor, x_cal, absor, 
                                                    C=params['C'], epsilon=params['epsilon'], 
                                                    gamma=params['gamma'], kernel=params['kernel'])
            
            diff_cal = y_cal_pred - x_cal
            # RMSE for each component
            RMSEcal[k_idx, :] = np.sqrt(np.mean(diff_cal**2, axis=0))
            
            # --- 2. Cross Validation ---
            pred_cv_accum = np.zeros((nd, nc))
            
            # Helper to generate CV splits
            for i in range(folds):
                if cv_type == 'venetian':
                    val_idx = np.arange(i, nd, folds)
                elif cv_type == 'random':
                     # For random, we usually pre-calculate folds, but here using the shuffled indices
                     start = i * fold_size
                     end = min((i+1)*fold_size, nd)
                     val_idx_indices = np.arange(start, end)
                     val_idx = indices[val_idx_indices]
                     val_idx = val_idx[val_idx < nd] # Safety
                else: # Sequential
                    start = i * fold_size
                    end = min((i+1)*fold_size, nd)
                    val_idx = np.arange(start, end)
                    val_idx = val_idx[val_idx < nd]

                mask = np.ones(nd, dtype=bool)
                mask[val_idx] = False
                train_idx = np.arange(nd)[mask]
                
                if len(val_idx) == 0: continue
                
                X_train_cv = absor[train_idx]
                Y_train_cv = x_cal[train_idx]
                X_val_cv = absor[val_idx]
                
                y_val_p = model_instance.fit_predict(X_train_cv, Y_train_cv, X_val_cv, 
                                                     C=params['C'], epsilon=params['epsilon'], 
                                                     gamma=params['gamma'], kernel=params['kernel'])
                pred_cv_accum[val_idx] = y_val_p
                
            diff_cv = pred_cv_accum - x_cal
            RMSECV[k_idx, :] = np.sqrt(np.mean(diff_cv**2, axis=0))
            
            # Restore units for logging
            RMSECV_conc[k_idx, :] = RMSECV[k_idx, :] * xmax
            RMSEcal_conc[k_idx, :] = RMSEcal[k_idx, :] * xmax

        # Save Results
        if output_dir:
            k_col = np.arange(1, kmax + 1).reshape(-1, 1)
            # Create a header of column names
            header = "k\t" + "\t".join(cname)
            np.savetxt(os.path.join(output_dir, 'Erro_cv.txt'), np.hstack([k_col, RMSECV_conc]), header=header, delimiter='\t')
            
            # Find best per component
            min_rmse = np.min(RMSECV_conc, axis=0) # shape (nc,)
            min_idx = np.argmin(RMSECV_conc, axis=0) # shape (nc,)
            
            # Print Best
            print("\n  --- SVR Best Parameters ---")
            for j in range(nc):
                best_i = min_idx[j]
                p = active_params[best_i]
                print(f"  {cname[j]}: RMSE={min_rmse[j]:.4f} | C={p['C']}, Eps={p['epsilon']}, Gam={p['gamma']}, Ker={p['kernel']}")
                
            # Saving minimos: format [min_rmse; min_index]
            # row 0: rmse values
            # row 1: index (1-based)
            np.savetxt(os.path.join(output_dir, 'minimos.txt'), np.vstack([min_rmse, min_idx+1]), delimiter='\t')
            
            # Save param list for lookup
            with open(os.path.join(output_dir, "params_log.txt"), "w") as f:
                for idx, p in enumerate(active_params):
                    f.write(f"{idx+1}\t{p}\n")

        return RMSECV, RMSECV_conc, RMSEcal, RMSEcal_conc
