import numpy as np
from ..utils import zscore_matlab_style

class PLS:
    def __init__(self):
        pass

    def nipals(self, X, Y, h):
        """
        Implements the NIPALS algorithm from pls.sci.
        Strict 1:1 translation of the mathematical operations.
        """
        m, n = X.shape
        # Ensure Y is 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        E = X.copy()
        F = Y.copy()
        
        ssX = np.sum(E**2)
        ssY = np.sum(F**2)
        
        # Initialize output matrices
        W = np.zeros((n, h))
        P = np.zeros((n, h))
        T = np.zeros((m, h))
        U = np.zeros((Y.shape[1], h)) # Placeholder for compatibility, shape might vary if Y has >1 col
        
        U_scores = np.zeros((m, h)) 
        Q_loadings = np.zeros((Y.shape[1], h)) 

        r2X = np.zeros(h)
        r2Y = np.zeros(h)

        for i in range(h):
            u = F[:, 0].reshape(-1, 1)
            uold = np.ones((m, 1)) * 100
            
            # Convergence loop
            while np.linalg.norm(uold - u) > 1e-5:
                uold = u.copy()
                w = E.T @ u
                w = w / np.linalg.norm(w)
                t = E @ w
                q = F.T @ t / (t.T @ t)
                u = F @ q / np.sqrt(q.T @ q)
            
            p = E.T @ t / (t.T @ t)
            
            # Store components
            W[:, i:i+1] = w
            P[:, i:i+1] = p
            T[:, i:i+1] = t
            U_scores[:, i:i+1] = u
            Q_loadings[:, i:i+1] = q
            
            # Deflation
            E = E - t @ p.T
            F = F - t @ q.T
            
            # Variance explained (Strict Scilab logic)
            # r2X(i) = 100*(t'*t*(p'*p)/ssX)
            r2X[i] = 100 * ((t.T @ t) * (p.T @ p) / ssX).item()
            r2Y[i] = 100 * ((t.T @ t) * (q.T @ q) / ssY).item()

        # Regression Coefficients Calculation: B = W * pinv(P' * W) * Q'
        # Scilab: B = W*pinv(P'*W)*Q';
        B = W @ np.linalg.pinv(P.T @ W) @ Q_loadings.T
        
        return B, T, P, U_scores, Q_loadings, W, r2X, r2Y

    def predict_model(self, X, Y, k, Xt=None, teste_switch=1):
        """
        Equivalent to pls_model.sci
        teste_switch: 0 or 1 (normalization mode)
        """
        # Ensure Inputs are 2D
        X = np.array(X)
        Y = np.array(Y)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        if Xt is not None: Xt = np.array(Xt)
        
        if teste_switch == 0:
            # Case 0: Normalize X independent of Xt
            Xnorm, Xmed, Xsig = zscore_matlab_style(X)
            Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
            
            B, T, _, _, _, _, _, _ = self.nipals(Xnorm, Ynorm, k)
            
            Ynormp = Xnorm @ B
            # Denormalize Yp
            Yp = Ynormp * Ysig + Ymed
            
            Ytp = None
            if Xt is not None:
                # Xt normalized using X's parameters
                Xtnorm = (Xt - Xmed) / Xsig
                Ytnormp = Xtnorm @ B
                Ytp = Ytnormp * Ysig + Ymed
                
            return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': B, 'T': T}

        elif teste_switch == 1:
            # Case 1: Normalize [X; Xt] together
            n = X.shape[0]
            if Xt is not None:
                Combined = np.vstack([X, Xt])
                Combined_norm, Xmed, Xsig = zscore_matlab_style(Combined)
                Xnorm = Combined_norm[:n, :]
                Xtnorm = Combined_norm[n:, :]
            else:
                # Fallback if no Xt provided but switch is 1
                Xnorm, Xmed, Xsig = zscore_matlab_style(X)
                Xtnorm = None
                
            Ynorm, Ymed, Ysig = zscore_matlab_style(Y)
            
            B, T, _, _, _, _, _, _ = self.nipals(Xnorm, Ynorm, k)
            
            Ynormp = Xnorm @ B
            Yp = Ynormp * Ysig + Ymed
            
            Ytp = None
            if Xt is not None and Xtnorm is not None:
                Ytnormp = Xtnorm @ B
                Ytp = Ytnormp * Ysig + Ymed
                
            return Yp, Ytp, {'Xmed': Xmed, 'Xsig': Xsig, 'Ymed': Ymed, 'Ysig': Ysig, 'Beta': B, 'T': T}
