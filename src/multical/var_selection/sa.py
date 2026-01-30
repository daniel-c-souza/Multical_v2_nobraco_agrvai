import numpy as np
from src.multical.var_selection.utils import calculate_rmsecv_fast

def run_simulated_annealing(absor, x, kmax, folds, cv_type, max_iter=100, initial_temp=0.1, alpha=0.90):
    """
    Performs Simulated Annealing for variable selection.
    """
    n_vars = absor.shape[1]
    
    # Start with a random subset (e.g. 50% of variables)
    current_mask = np.random.rand(n_vars) > 0.5
    if np.sum(current_mask) < 2: current_mask[:2] = True
    
    # Initial Cost
    absor_sub = absor[:, current_mask]
    # We only need the scalar RMSE
    current_rmse, _, _ = calculate_rmsecv_fast(absor_sub, x, kmax, folds, cv_type)
    
    best_mask = current_mask.copy()
    best_rmse = current_rmse
    
    T = initial_temp
    
    print(f"SA Initialization: RMSE={current_rmse:.4f}, Vars={np.sum(current_mask)}, T={T}")
    
    for i in range(max_iter):
        # Create Neighbor: Flip state of some variables
        # Mix of small moves (1 flip) and larger moves
        if np.random.rand() < 0.7:
             n_change = 1
        else:
             n_change = max(2, int(0.01 * n_vars))
             
        idx_change = np.random.choice(n_vars, n_change, replace=False)
        
        neighbor_mask = current_mask.copy()
        neighbor_mask[idx_change] = ~neighbor_mask[idx_change]
        
        if np.sum(neighbor_mask) < 2: continue
        
        # Evaluate Neighbor
        absor_neigh = absor[:, neighbor_mask]
        neigh_rmse, _, _ = calculate_rmsecv_fast(absor_neigh, x, kmax, folds, cv_type)
        
        # Metastability / Acceptance
        delta = neigh_rmse - current_rmse
        
        accept = False
        if delta < 0:
            accept = True
        else:
            # Boltzmann
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                accept = True
        
        if accept:
            current_mask = neighbor_mask
            current_rmse = neigh_rmse
            # Is it global best?
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_mask = current_mask.copy()
                print(f"  [Iter {i+1}] New Best: RMSE={best_rmse:.4f} (Vars: {np.sum(best_mask)})")
        
        # Cool down
        T = T * alpha
        
    return best_mask, best_rmse
