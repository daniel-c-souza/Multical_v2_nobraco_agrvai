import numpy as np

def zscore_matlab_style(X, flag=0):
    """
    Computes ztest similar to Scilab/Matlab's zscore.
    Uses ddof=1 (N-1) for standard deviation by default (flag=0).
    If flag=1, uses ddof=0 (N).
    
    Returns:
        Xconst: standardizes data
        mu: mean
        sigma: standard deviation
    """
    # Ensure X is at least 1D, but logic usually implies 2D (samples x vars) or 1D vector
    X = np.array(X)
    
    mu = np.mean(X, axis=0)
    # flag=0 implies N-1 (sample std), which is default in Scilab/Matlab
    ddof = 1 if flag == 0 else 0
    sigma = np.std(X, axis=0, ddof=ddof)
    
    # Avoid division by zero for constant columns
    # Replicate Scilab/Matlab behavior: 0/0 -> NaN, x/0 -> Inf usually
    # But usually we want to avoid crashing. 
    # If sigma is 0, the feature is constant. (X-mu) is 0. 0/0.
    # We will replace sigma=0 with 1 to keep 0s.
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe == 0] = 1.0
    
    Xconst = (X - mu) / sigma_safe
    
    return Xconst, mu, sigma

def polyfit_matlab_style(x, y, n, w=None):
    """
    Wrapper for numpy.polyfit to behave like Scilab/Matlab's polyfit.
    Specifically regarding return values and weighting.
    
    Scilab: [p, xt] = polyfit(x, y, n, w)
    
    In Scilab/Matlab, polyfit returns coefficients in decreasing power.
    Numpy also returns decreasing power.
    
    The 'w' in Scilab multiplies the equation, effectively minimizing sum(w^2 * err^2).
    Numpy 'w' argument also multiplies the residuals, effectively minimizing sum(w^2 * err^2).
    So they are compatible.
    
    Scilab supports normalization (output xt). We won't implement that unless needed.
    """
    if w is not None:
        p = np.polyfit(x, y, n, w=w)
    else:
        p = np.polyfit(x, y, n)
    
    return p
