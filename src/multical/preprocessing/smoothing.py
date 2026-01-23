import numpy as np

def alisar(alis, absor, lambda_, raio):
    """
    Implements 'alisar.sci' smoothing logic.
    
    Arguments:
        alis: 1 (Moving Average) or 2 (Downsampling MA)
        absor: Data matrix (samples x wavelengths) or vector
        lambda_: Wavelengths vector
        raio: Radius (half-window). Window size = 2*raio + 1
        
    Returns:
        Aalis: Smoothed data
        lambalis: New wavelengths (same if alis=1, reduced if alis=2)
    """
    # Ensure inputs are correct shape
    A = np.array(absor)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    # Store dimensions
    nd, nl = A.shape
    
    lambalis = np.array(lambda_)
    
    if alis == 1:
        # Case 1: Number of data points remains the same
        # Logic:
        # Edges (Start): for i=1:raio, window is [1, i+raio] (1-based)
        # i.e., Python 0 to i+raio
        # Edges (End): for i=nl-raio+1:nl, window is [i-raio, nl]
        
        Aalis = np.zeros_like(A)
        
        # Scilab loops 1 to raio (indexes 0 to raio-1)
        for i in range(raio):
            # Scilab: mean(A(:, 1:i+raio)) -> Python: 0 : i+raio+1
            # Wait. Scilab i=1 => mean 1:1+raio (indices 0 to raio)
            # Python i (0..raio-1). 
            # Corresp Scilab i = i_py + 1.
            # Scilab Window: 1 : (i_py+1)+raio -> Python indices 0 : i_py+1+raio
            
            # Scilab code: for i=1:raio, mean(A(:,1:i+raio))
            # i=1: 1:1+raio
            # i=raio: 1:2*raio
            
            # Python i goes 0 to raio-1.
            # Effective Scilab i = i + 1.
            # End index (exclusive) = i + 1 + raio.
            Aalis[:, i] = np.mean(A[:, 0 : i + 1 + raio], axis=1)
            
        # Middle: i = 1+raio : nl-raio (Scilab 1-based)
        # Python indices: raio : nl-raio
        # Window: i-raio : i+raio (Scilab) -> Python [i-raio : i+raio+1]
        for i in range(raio, nl - raio):
            Aalis[:, i] = np.mean(A[:, i - raio : i + raio + 1], axis=1)
            
        # End: i = nl-raio+1 : nl (Scilab) -> Python indices nl-raio : nl
        # Window: i-raio : nl (Scilab) -> Python i-raio : nl
        for i in range(nl - raio, nl):
            Aalis[:, i] = np.mean(A[:, i - raio : nl], axis=1)
            
        return Aalis, lambalis

    elif alis == 2:
        # Case 2: Downsampling
        nalis = int(nl / (2 * raio + 1))
        Aalis = np.zeros((nd, nalis))
        lambalis_new = np.zeros(nalis)
        
        # j = raio+1 (Scilab 1-based) -> Python index raio
        j = raio
        
        for i in range(nalis):
            # Scilab loop i=1:nalis. i here is 0:nalis-1
            # Scilab: mean(A(:, j-raio : j+raio))
            # Python: j-raio : j+raio+1
            
            Aalis[:, i] = np.mean(A[:, j - raio : j + raio + 1], axis=1)
            lambalis_new[i] = lambalis[j]
            
            j = j + 2 * raio + 1
            
        return Aalis, lambalis_new
    
    else:
        print('Alisamento não realizado')
        return A, lambalis
