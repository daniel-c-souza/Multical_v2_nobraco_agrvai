import numpy as np
from multiprocessing import Pool, cpu_count
from src.multical.var_selection.utils import pso_worker_wrapper

def run_pso(absor, x, kmax, folds, cv_type, n_particles=30, max_iter=50, w=0.9, c1=1.49, c2=1.49):
    """
    Performs Particle Swarm Optimization (PSO) for variable selection with Parallel Batch Processing.
    Uses a binary PSO approach where position [0,1] is probability of inclusion.
    """
    n_vars = absor.shape[1]
    
    # Initialize Particles
    # X: Position (Probability 0..1)
    # V: Velocity
    X_pos = np.random.rand(n_particles, n_vars)
    V_vel = np.random.uniform(-1, 1, (n_particles, n_vars))
    
    P_best_pos = X_pos.copy()
    P_best_score = np.full(n_particles, np.inf)
    
    G_best_pos = np.zeros(n_vars)
    G_best_score = np.inf
    G_best_mask = np.ones(n_vars, dtype=bool)

    print(f"PSO Initialization: {n_particles} particles, {max_iter} iterations")
    
    # Determine parallel power
    try:
        n_cpu = max(1, cpu_count() - 1)
    except:
        n_cpu = 1
        
    print(f"Parallel Execution on {n_cpu} cores.")
    
    # Initialize Pool ONCE
    with Pool(processes=n_cpu) as pool:
        for iteration in range(max_iter):
            
            # Linear Decay of Inertia Weight (w)
            w_curr = w - ((w - 0.4) * (iteration / max_iter))
            
            # 1. Generate all masks for this iteration
            masks_to_eval = []
            for i in range(n_particles):
                # Sigmoid transfer
                sigmoid_x = 1 / (1 + np.exp(-X_pos[i]))
                current_mask = np.random.rand(n_vars) < sigmoid_x
                masks_to_eval.append(current_mask)
            
            # 2. Evaluate in Parallel
            # Create args list. Note: Absor/X are reused, Copy-on-Write handles memory on Linux
            args_list = [(m, absor, x, kmax, folds, cv_type) for m in masks_to_eval]
            
            # This blocks until all particles are evaluated
            scores = pool.map(pso_worker_wrapper, args_list)
            
            # 3. Update Swarm Logic
            for i in range(n_particles):
                score = scores[i]
                current_mask = masks_to_eval[i]
                
                # Update Personal Best
                if score < P_best_score[i]:
                    P_best_score[i] = score
                    P_best_pos[i] = X_pos[i].copy()
                    
                    # Update Global Best
                    if score < G_best_score:
                        G_best_score = score
                        G_best_pos = X_pos[i].copy()
                        G_best_mask = current_mask.copy()
                        print(f"  [Iter {iteration+1}] New Global Best: RMSE={G_best_score:.4f} (Vars: {np.sum(G_best_mask)}) w={w_curr:.2f}")
            
            # 4. Update Velocity and Position
            r1 = np.random.rand(n_particles, n_vars)
            r2 = np.random.rand(n_particles, n_vars)
            
            V_vel = (w_curr * V_vel) + (c1 * r1 * (P_best_pos - X_pos)) + (c2 * r2 * (G_best_pos - X_pos))
            V_vel = np.clip(V_vel, -4, 4)
            X_pos = X_pos + V_vel
        
    return G_best_mask, G_best_score
