import numpy as np
from ..utils import zscore_matlab_style

def spa_clean(X, cini, nk):
    """
    Successive Projections Algorithm.
    Arguments:
        X: Data matrix (samples x variables)
        cini: Initial variable index (0-based)
        nk: Number of variables to select
    Returns:
        ind: Selected indices (0-based)
    """
    X = np.array(X)
    m, n = X.shape
    Xn = np.zeros_like(X, dtype=float)
    
    # Normalize columns to norm 1
    for i in range(n):
        nm = np.linalg.norm(X[:, i])
        if nm > 0:
            Xn[:, i] = X[:, i] / nm
            
    # List of original indices available in Xn
    indices = list(range(n))
    
    # Output array
    ind = np.zeros(nk, dtype=int)
    ind[0] = cini
    
    # xant logic: vector of the previously selected variable
    # Extract column corresponding to cini
    # IMPORTANT: We must find where 'cini' is in our current tracked 'indices'
    # Initially they map 1:1.
    curr_idx_in_array = indices.index(cini) 
    
    xant = Xn[:, curr_idx_in_array].copy().reshape(-1, 1)
    
    # Remove selected column from Xn and indices list
    Xn = np.delete(Xn, curr_idx_in_array, axis=1)
    indices.pop(curr_idx_in_array)
    
    for ik in range(1, nk):
        # Project remaining columns onto orthogonal complement of xant
        # Pxj = xj - (xj'*xant)*xant * inv(xant'*xant)
        
        # Matrix operation for: for j in cols: vector projection
        # Proj = Xn - xant @ (xant.T @ Xn) / (xant.T @ xant)
        
        denom = (xant.T @ xant).item()
        if denom < 1e-15: denom = 1e-15 # Avoid divide by zero
        
        coeffs = (xant.T @ Xn) / denom
        Xn = Xn - (xant @ coeffs)
        
        # Find column with max norm
        norms = np.linalg.norm(Xn, axis=0)
        indmax = np.argmax(norms)
        
        # Save original index
        ind[ik] = indices[indmax]
        
        # Update xant for next iteration (the newly projected vector with max norm)
        xant = Xn[:, indmax].copy().reshape(-1, 1)
        
        # Remove selected
        Xn = np.delete(Xn, indmax, axis=1)
        indices.pop(indmax)
        
    return ind

def spa_model(X, Y, k, cini, Xt=None, teste_switch=1):
    """
    Equivalent to spa_model.sci
    cini: 0-based index
    """
    X = np.array(X)
    Y = np.array(Y)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    if Xt is not None: Xt = np.array(Xt)

    # ---------------- Pre-processing Logic (Switch) ----------------
    if teste_switch == 0:
        Xnorm, Xmed, Xsig = zscore_matlab_style(X)
        Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
        Xtnorm = None
        if Xt is not None:
            Xtnorm = (Xt - Xmed) / Xsig

    elif teste_switch == 1:
        n = X.shape[0]
        if Xt is not None:
            Combined = np.vstack([X, Xt])
            Combined_norm, Xmed, Xsig = zscore_matlab_style(Combined)
            Xnorm = Combined_norm[:n, :]
            Xtnorm = Combined_norm[n:, :]
        else:
            Xnorm, Xmed, Xsig = zscore_matlab_style(X)
            Xtnorm = None
            
        Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
        
    # ---------------- SPA Logic ----------------

    
    ind = spa_clean(X, cini, k)
    
    # OLS on selected columns of Xnorm
    Xsel = Xnorm[:, ind]
    
    Beta, _, _, _ = np.linalg.lstsq(Xsel, Ynorm, rcond=None)
    
    Ynormp = Xsel @ Beta
    Yp = Ynormp * Ysig + Ymed
    
    Ytp = None
    if Xt is not None and Xtnorm is not None:
        Xtsel = Xtnorm[:, ind]
        Ytnormp = Xtsel @ Beta
        Ytp = Ytnormp * Ysig + Ymed
        
    return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': Beta, 'SelectedVars': ind}
