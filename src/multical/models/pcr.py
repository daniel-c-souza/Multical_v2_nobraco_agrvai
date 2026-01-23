import numpy as np
from ..utils import zscore_matlab_style

def pcr_model(X, Y, k, Xt=None, teste_switch=1):
    """
    Equivalent to pcr_model.sci
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
    
    # ---------------- PCR Algorithm ----------------
    # Scilab: [eigvec,eigval] = spec(Xnorm'*Xnorm);
    # Scilab 'spec' usually returns ascending eigenvalues.
    # PCR requires largest eigenvalues (Principal Components).
    # Numpy eigh returns ascending. We must reverse to get Top K.
    
    Cov = Xnorm.T @ Xnorm
    eigval, eigvec = np.linalg.eigh(Cov)
    
    # Sort descending
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, idx]
    eigval = eigval[idx]
    
    # Select k components
    eigvec_k = eigvec[:, :k]
    
    # Scores
    Xeig = Xnorm @ eigvec_k
    
    # Regression (Least Squares): Beta = Xeig \ Ynorm
    # Beta here connects Scores -> Y
    # np.linalg.lstsq returns (x, residuals, rank, s)
    Beta, _, _, _ = np.linalg.lstsq(Xeig, Ynorm, rcond=None)
    
    # Prediction on Calibration
    Ynormp = Xeig @ Beta
    Yp = Ynormp * Ysig + Ymed
    
    Ytp = None
    if Xt is not None and Xtnorm is not None:
        # Prediction on Test
        # Xteig = Xtnorm * eigvec(:,1:k); // scores
        Xteig = Xtnorm @ eigvec_k
        Ytnormp = Xteig @ Beta
        Ytp = Ytnormp * Ysig + Ymed
        
    return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': Beta, 'T': Xeig}
