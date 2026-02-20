import numpy as np

def snv(absor):
    """
    Standard Normal Variate (SNV) transformation.
    x_snv = (x - mean(x)) / std(x) for each spectrum (row).
    """
    absor = np.array(absor, dtype=float)
    if absor.ndim == 1:
        absor = absor.reshape(1, -1)
        
    mean_vec = np.mean(absor, axis=1, keepdims=True)
    std_vec = np.std(absor, axis=1, ddof=1, keepdims=True)
    
    # Avoid division by zero
    std_vec[std_vec == 0] = 1.0
    
    return (absor - mean_vec) / std_vec

def emsc(absor, lambda_, degree=2, reference=None):
    """
    Extended Multiplicative Scatter Correction (EMSC).
    Corrects baseline and scaling effects using a reference spectrum and polynomial terms.
    
    Parameters:
    - absor: Input spectra (n_samples, n_wavelengths)
    - lambda_: Wavelengths array (n_wavelengths,) or None. 
               If None, indices are used.
    - degree: Polynomial degree for baseline (0=MSC (offset), 1=linear, 2=quadratic)
    - reference: Reference spectrum (n_wavelengths,). If None, mean of absor is used.
    
    Returns:
    - corrected_absor
    """
    absor = np.array(absor, dtype=float)
    if absor.ndim == 1:
        absor = absor.reshape(1, -1)
        
    n_samples, n_points = absor.shape
    
    if lambda_ is None:
        lambda_ = np.arange(n_points)
    else:
        lambda_ = np.array(lambda_, dtype=float)
        
    if reference is None:
        reference = np.mean(absor, axis=0)
    
    # Make sure reference is same shape
    if reference.shape[0] != n_points:
         raise ValueError(f"Reference spectrum shape {reference.shape} does not match wavelengths {n_points}")

    # Center wavelengths for numerical stability
    l_centered = lambda_ - np.mean(lambda_)
    
    # Construct Design Matrix columns: [Reference, 1, lambda, lambda^2, ...]
    X_cols = [reference]
    
    # Bias (degree >= 0 is typical for MSC)
    # Standard MSC usually includes offset (a) -> degree=0 polynomial (x^0)
    X_cols.append(np.ones(n_points))
    
    for i in range(1, degree + 1):
        X_cols.append(l_centered**i)
        
    X_matrix = np.vstack(X_cols).T # (n_points, 2 + degree)
    
    # Solve X_matrix @ Coeffs = absor.T
    # We want to find [b, a, d, e, ...] for each sample
    # lstsq expects (M, N) @ (N, K) = (M, K)
    # Here M=n_points, N=(2+degree), K=n_samples
    
    coeffs, _, _, _ = np.linalg.lstsq(X_matrix, absor.T, rcond=None)
    
    # coeffs shape: (2+degree, n_samples)
    coeffs = coeffs.T # (n_samples, 2+degree)
    
    # Extract coefficients
    b = coeffs[:, 0:1] # multiplicative factor (vector)
    
    # Polynomial coeffs (including offset)
    poly_coeffs = coeffs[:, 1:] 
    X_poly = X_matrix[:, 1:] # (n_points, 1+degree)
    
    # Calculate polynomial background
    # (n_samples, 1+degree) @ (1+degree, n_points) -> (n_samples, n_points)
    background = poly_coeffs @ X_poly.T
    
    # Prevent division by zero or negative flip if b < 0 (unlikely but possible)
    # Usually we don't enforce b > 0 in basic implementation but it's good to be aware.
    mask_small = np.abs(b) < 1e-6
    b[mask_small] = 1.0 # fallback
    
    corrected_absor = (absor - background) / b
    
    return corrected_absor
