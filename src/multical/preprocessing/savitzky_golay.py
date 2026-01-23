import numpy as np

def sgolay_filt(x, y, n, F, d=0):
    """
    Implements 'sgolay_filt.sci'
    
    Arguments:
        x: Independent variable (vector) - Wavelengths
        y: Dependent variable (vector) - Absorbance (Warning: Scilab code treats y as column vector)
           If y is a matrix, this function should iterate? 
           The original Scilab code 'sgolay_filt' takes 'y' as vector.
           The 'func_pretreatment' calls it in a loop: 
           'for i=1:nd absortemp(i,:)=sgolay_filt(lambda',absor(i,:)',ordem,janela,der_ordem)';'
    
    This function processes a single signal y (1D equivalent).
    
    n: Order of polynomial
    F: Window size (must be odd usually, but code handles even?)
    d: Derivative order
    """
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    N = len(x)
    
    # Check linearity
    dx = x[1:] - x[:-1]
    # Check if all dx are close
    if not np.allclose(dx, dx[0]):
        print("x must be linearly spaced")
        # In strictly 1:1 port, we might abort or continue. 
        # Scilab code prints and continues but the logic relies on Dx.
    
    if F < n + 1:
        n = F - 1
        print("order reduction due to window size")
        
    # Scaling setup based on FIRST window
    Dx = x[F-1] - x[0]
    dxrdx = 2.0 / Dx
    
    xr = np.linspace(-1, 1, F)
    
    # Vandermonde Matrix X
    # X columns: 1, xr, xr^2, ... xr^n
    X_mat = np.zeros((F, n + 1))
    for i in range(n + 1):
        X_mat[:, i] = xr**i
        
    # Projection Matrix S_uni calculation
    # S_uni = D * inv(X'X) * X'
    # Where D is differentiation operator on the coefficients
    
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    
    if d == 0:
        S_uni = X_mat @ XtX_inv @ X_mat.T
    else:
        # Derivative Matrix dX applied to coefficients?
        # Scilab Logic:
        # dX constructed from X by differentiating columns
        # Column j (xr^j) -> j*xr^(j-1)
        
        # dX in Scilab code:
        # dX = [zeros(F,1) X(:,1:n)] -> Shifts columns right, fills first with 0 (since deriv const is 0)
        # then dX(:,i+1) = i*dX(:,i+1). 
        # Let's trace i=2:n (Scilab indices 1-based, matrix size n+1 columns).
        # Col 1 (x^0) -> 0
        # Col 2 (x^1) -> 1 * x^0
        # Col 3 (x^2) -> 2 * x^1
        
        current_mat = X_mat.copy()
        
        for k in range(d):
            # Differentiate current_mat columns
            # Shift right
            next_mat = np.zeros_like(current_mat)
            next_mat[:, 1:] = current_mat[:, :-1]
            
            # Multiply by powers
            # Column i (index i) corresponds to power i?
            # Initial X_mat logic: X[:, i+1] = xr^i in Scilab (i=1..n => powers 1..n). Col 1 is power 0.
            # Python Col i is power i.
            # D(x^p) = p * x^(p-1).
            # So NewCol i = p * OldCol i? No.
            # NewCol corresponding to power p (at index p) comes from OldCol at index p?
            # Wait.
            # Col 0: 1. Deriv: 0.
            # Col 1: x. Deriv: 1 (Col 0).
            # Col 2: x^2. Deriv: 2x (2 * Col 1).
            # Col p: x^p. Deriv: p * x^(p-1) (p * Col p-1).
            
            # So: NextMat[:, p] = p * CurrentMat[:, p-1] (for p >= 1).
            
            # Scilab code:
            # dX = [zeros(F,1) X(:,1:n)]
            # for i=2:n (cols 3 to n+1? No, i is power?)
            # Scilab loop `for i=2:n` refers to power `i`?
            # `X(:,i+1) = xr.^i`. So col `i+1` is power `i`.
            # `dX(:,i+1) = i*dX(:,i+1)`. 
            # `dX` was shifted `X`. `dX(:,i+1)` contains `X(:,i)` which is `xr^(i-1)`.
            # So `dX(:,i+1)` becomes `i * xr^(i-1)`.
            # This is derivative of `xr^i`. Correct.
            
            temp_mat = np.zeros_like(current_mat)
            for p in range(1, n + 1):
                temp_mat[:, p] = p * current_mat[:, p-1]
            current_mat = temp_mat
            
        D_mat = current_mat
        S_uni = D_mat @ XtX_inv @ X_mat.T

    # Apply Filtering
    raio = (F - 1) // 2
    yfilt = y.copy()
    
    # 1. First Window (up to raio)
    # yfilt(1:raio+1) = S_uni(1:raio+1,:)*y(1:F)
    # Python: yfilt[0 : raio+1]
    yfilt[0 : raio + 1] = S_uni[0 : raio + 1, :] @ y[0 : F]
    
    # 2. Middle
    # yfilt(i) = S_uni(raio+1,:)*y(i-raio:i+raio)
    # Scilab i from raio+2 to N-raio-1
    # Python i from raio+1 to N-raio-2?
    # Scilab i indexes reference the OUTPUT point.
    # Scilab: i=raio+2. Window center.
    # Python corresponding index: raio+1.
    # Corresponds to window starting at (raio+1)-raio = 1 (Python 1).
    # Wait. If window is size F=2*r+1. Center is at r.
    # First valid center is at index r. (0..r..2r).
    # Scilab logic: for i=raio+2 ...
    # i=raio+2 (1-based) is index raio+1 (0-based).
    # Window y(i-raio : i+raio).
    # If i_py = raio+1. i_sc = raio+2.
    # Win_sc = 2 : 2+2r.
    # Win_py = 1 : 1+2r+1 (indices).
    
    # Range of i_py:
    # Start: raio + 1.
    # End: N - raio - 2 (exclusive: N - raio - 1). ??
    # Scilab end: N-raio-1. (index N-raio-2).
    # Python loop: range(raio + 1, N - raio) ??
    # Let's check last point.
    # Scilab Last: N-raio-1.
    # Python Index: N-raio-2.
    # Python Range end (exclusive) should be N-raio-1.
    
    middle_indices = range(raio + 1, N - raio)
    # Wait. N-raio-1 (Scilab) -> N-raio-2 (Python).
    # range(start, end) includes start, excludes end.
    # range(raio+1, N-raio-1). 
    # Let's tracing.
    
    for i in range(raio + 1, N - raio):
        # Window start: i - raio
        # Window end: i + raio + 1
        window = y[i - raio : i + raio + 1]
        
        # Center row of S_uni: index raio
        yfilt[i] = S_uni[raio, :] @ window
        
    # 3. Last Window
    # yfilt(N-raio:N) = S_uni(raio+1:F,:)*y(N-F+1:N)
    # Python indices: N-raio-1 : N
    # Scilab N-raio is Python N-raio-1.
    
    yfilt[N - raio - 1 : N] = S_uni[raio : F, :] @ y[N - F : N]
    
    # Scaling
    if d != 0:
        yfilt = yfilt * (dxrdx**d)
        
    return yfilt
