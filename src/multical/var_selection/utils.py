import numpy as np
from src.multical.models.pls import PLS
from src.multical.utils import zscore_matlab_style
from scipy.stats import f as f_dist

def calculate_rmsecv_fast(absor, x, kmax, folds=5, cv_type='venetian'):
    """
    Optimized RMSECV calculation that runs NIPALS once per fold up to kmax.
    drastically faster than re-running PLS for each k.
    Supports PLS2 (Multi-Y).
    """
    # Normalize X (target Y for PLS)
    xmax = np.max(x, axis=0) # (3,)
    xmax[xmax == 0] = 1
    x_norm = x / xmax
    
    nd, nl = absor.shape
    ny = x.shape[1] # Number of Y variables
    
    # CV Indices
    if cv_type == 'random':
        indices = np.random.permutation(nd)
    else:
        indices = np.arange(nd) # Sequential
    
    fold_size = int(np.ceil(nd / folds))
    
    # Storage for Sum of Squared Errors: (kmax,)
    sse_all = np.zeros(kmax)
    
    model = PLS()
    
    for i in range(folds):
        if cv_type == 'venetian':
            val_idx = np.arange(i, nd, folds)
        else:
            s = i * fold_size
            e = min((i+1)*fold_size, nd)
            val_idx_raw = np.arange(s, e)
            val_idx_raw = val_idx_raw[val_idx_raw < nd]
            val_idx = indices[val_idx_raw]
        
        if len(val_idx) == 0: continue
        
        mask = np.ones(nd, dtype=bool)
        mask[val_idx] = False
        train_idx = np.arange(nd)[mask]
        
        # Data Split
        X_train_raw = absor[train_idx, :]
        Y_train_raw = x_norm[train_idx, :] 
        X_val_raw = absor[val_idx, :]
        Y_val_target = x_norm[val_idx, :]
        
        # --- Normalization (Switch=1 Logic: Normalize [Train; Val] together) ---
        Combined_X = np.vstack([X_train_raw, X_val_raw])
        Combined_X_norm, Xmed, Xsig = zscore_matlab_style(Combined_X)
        
        n_train = X_train_raw.shape[0]
        X_train = Combined_X_norm[:n_train, :]
        X_val = Combined_X_norm[n_train:, :]
        
        # Y Normalization (Train Only)
        Y_train, Ymed, Ysig = zscore_matlab_style(Y_train_raw)
        
        # --- Run NIPALS ONCE ---
        _, _, P_all, _, Q_all, W_all, _, _ = model.nipals(X_train, Y_train, kmax)
        
        # --- Incremental Prediction ---
        for k in range(1, kmax + 1):
             wk = W_all[:, :k]
             pk = P_all[:, :k]
             qk = Q_all[:, :k]
             
             # Beta = W * inv(P.T * W) * Q.T
             pw = pk.T @ wk
             # Use pinv for safety
             pw_inv = np.linalg.pinv(pw)
             
             Beta_k = wk @ pw_inv @ qk.T
             
             # Predict
             Y_val_pred_norm = X_val @ Beta_k
             
             # Denormalize
             # Ysig shape (1, 3), Ymed (1, 3)
             Y_val_pred = Y_val_pred_norm * Ysig + Ymed
             
             # SSE Aggregate
             # Difference from Y_val_target (which is x_norm[val_idx])
             diff = Y_val_pred - Y_val_target
             sse_all[k-1] += np.sum(diff**2)

    # Calculate RMSECV for all k
    # RMSE = sqrt(Mean(SSE))
    # Mean is divided by Total Elements Predicted (N samples * N_Y Variables)
    total_elements = nd * ny
    rmsecv_k = np.sqrt(sse_all / total_elements)
    
    # Scale back to original units average
    rmsecv_k = rmsecv_k * np.mean(xmax)
        
    return np.min(rmsecv_k), np.argmin(rmsecv_k) + 1, rmsecv_k

def select_k_ftest(RMSECV, n_cal):
    """
    Selects the optimal k using the Osten F-test logic:
    Finds the highest k that shows significant improvement.
    """
    kmax = len(RMSECV)
    best_k = 1
    
    # Iterate through steps k -> k+1
    for k_chk in range(kmax - 1):
        k_val = k_chk + 1
        
        rmse_sq_k = RMSECV[k_chk]**2
        rmse_sq_k_plus_1 = RMSECV[k_chk + 1]**2
        
        if rmse_sq_k_plus_1 == 0: eps = 1e-10
        else: eps = 0
        
        df2 = n_cal - k_val - 1
        if df2 <= 0: df2 = 1
        
        numerator = rmse_sq_k - rmse_sq_k_plus_1
        if numerator < 0:
            F_stat = 0
        else:
            F_stat = (numerator / (rmse_sq_k_plus_1 + eps)) * df2
            
        f_crit_val = f_dist.ppf(0.95, 1, df2)
        
        if F_stat >= f_crit_val:
            # Significant improvement k -> k+1
            # Check if this new target (k+1) is higher than current best
            if (k_val + 1) > best_k:
                best_k = k_val + 1
                
    return best_k

def pso_worker_wrapper(args):
    # Unpack arguments
    mask, absor, x, kmax, folds, cv_type = args
    if np.sum(mask) < 2: return 1e9
    absor_sub = absor[:, mask]
    rmse, _, _ = calculate_rmsecv_fast(absor_sub, x, kmax, folds, cv_type)
    return rmse
