import numpy as np

def loess(x, y, alpha, order=2):
    """
    LOESS smoother: locally fitted polynomials 
    
    Parameters:
    x (np.ndarray): Independent variable (e.g. wavelengths)
    y (np.ndarray): Dependent variable (e.g. absorbance)
    alpha (float): Smoothing parameter (0 < alpha <= 1)
    order (int): Polynomial order (1 or 2)
    
    Returns:
    np.ndarray: Smoothed y
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    
    # Validation
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    
    # Scale x to avoid ill-conditioning
    x_mean = np.mean(x)
    x_range = np.max(x) - np.min(x)
    if x_range == 0:
        return y
        
    x_scaled = (x - x_mean) / x_range
    
    y_out = np.zeros_like(y)
    
    # Window width
    # q = min(max(floor(alpha*n),order+3),n);
    q = int(min(max(np.floor(alpha * n), order + 3), n))
    
    for i in range(n):
        xi = x_scaled[i]
        deltax = np.abs(xi - x_scaled)
        
        # Sort or partition to find q-th element
        # We need the q-th smallest distance. 
        # 0-indexed: q-1
        if q >= n:
            qthdeltax = np.max(deltax)
        else:
            partitioned = np.partition(deltax, q-1)
            qthdeltax = partitioned[q-1]
            
        # Weight calculation
        # arg = min(deltax/(qthdeltax*max(alpha,1)),1);
        denom = qthdeltax * max(alpha, 1.0)
        
        if denom == 0:
            arg = np.zeros_like(deltax) # Should not happen if q > 1
        else:
            arg = np.minimum(deltax / denom, 1.0)
            
        weight = (1.0 - np.abs(arg)**3)**3
        
        # Select points with nonzero weights
        indices = np.where(weight > 1e-10)[0]
        
        if len(indices) > order:
            # Weighted polyfit
            
            coeffs = np.polyfit(x_scaled[indices], y[indices], order, w=weight[indices])
            y_out[i] = np.polyval(coeffs, xi)
        else:
            if len(indices) > 0:
                y_out[i] = np.mean(y[indices])
            else:
                y_out[i] = y[i]
                
    return y_out
