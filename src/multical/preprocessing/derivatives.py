import numpy as np

def diffmeu(A, lambda_):
    """
    Implements 'diffmeu.sci' derivatives.
    
    Arguments:
        A: Data matrix (samples x wavelengths)
        lambda_: Wavelengths vector
        
    Returns:
        dA: First derivative
        d2A: Second derivative
    """
    A = np.array(A)
    # Ensure 2D
    if A.ndim == 1:
        A = A.reshape(1, -1)
        
    dA = np.zeros_like(A)
    d2A = np.zeros_like(A)
    
    # Store dimensions
    n_samples, J = A.shape
    
    # Python indices: 0 to J-1
    # Scilab 1 => Python 0
    # Scilab J => Python J-1
    
    # --- First Derivative ---
    
    # Left Edge
    delta_start = lambda_[1] - lambda_[0]
    dA[:, 0] = (A[:, 1] - A[:, 0]) / delta_start
    
    # Right Edge
    delta_end = lambda_[J-1] - lambda_[J-2]
    dA[:, J-1] = (A[:, J-1] - A[:, J-2]) / delta_end
    
    # Central (Interior)
    # Scilab: delta = lambda(3:J)-lambda(1:J-2); 
    # Python: lambda[2:J] - lambda[0:J-2]
    delta_vec = lambda_[2:J] - lambda_[0:J-2]
    
    # Scilab Loop j=2:J-1 (Indices 1 to J-2)
    # Vectorized implementation
    # A(:, j+1) -> A[:, 2:J]
    # A(:, j-1) -> A[:, 0:J-2]
    
    # Verify shapes:
    # A[:, 2:J] is (samples, J-2)
    # delta_vec is (J-2,)
    
    dA[:, 1:J-1] = (A[:, 2:J] - A[:, 0:J-2]) / delta_vec
    
    # --- Second Derivative ---
    
    # Note: Uses FIXED delta based on first interval
    delta_fixed = lambda_[1] - lambda_[0]
    delta_sq = delta_fixed**2
    
    # Edges from dA
    d2A[:, 0] = (dA[:, 1] - dA[:, 0]) / delta_fixed
    d2A[:, J-1] = (dA[:, J-1] - dA[:, J-2]) / delta_fixed
    
    # Interior from A
    # Formula: (A(j+1) - 2A(j) + A(j-1)) / delta^2
    # A[:, 2:J] - 2*A[:, 1:J-1] + A[:, 0:J-2]
    
    term = A[:, 2:J] - 2 * A[:, 1:J-1] + A[:, 0:J-2]
    d2A[:, 1:J-1] = term / delta_sq
    
    return dA, d2A
