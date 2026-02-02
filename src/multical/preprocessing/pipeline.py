import numpy as np
import matplotlib.pyplot as plt
import os
from .smoothing import alisar
from .derivatives import diffmeu
from .savitzky_golay import sgolay_filt
from .loess import loess
from ..utils import zscore_matlab_style


def apply_pretreatment(pretreat_list, absor, lambda_, plot=True, output_dir=None, prefix=""):
    """
    Applies a sequence of pretreatment steps.
    """
    absor = np.array(absor, dtype=float)
    lambda_ = np.array(lambda_, dtype=float)
    
    # Ensure absor is 2D
    if absor.ndim == 1:
        absor = absor.reshape(1, -1)

    if plot:
        fig = plt.figure()
        plt.plot(lambda_, absor.T)
        plt.title(f"{prefix}Original Data")
        plt.xlabel("Wavelength")
        plt.ylabel("Absorbance")
        # plt.show(block=False)
        if output_dir:
            fig.savefig(os.path.join(output_dir, f'{prefix}Pretreatment_0_Original.png'))
        
    for i, step in enumerate(pretreat_list):
        method = step[0]
        print(f"Applying {method}")
        
        # Scilab often has 'plot' flag as last element
        # We will check if the last element is 1 (True) or 0 (False)
        # But we also have a global 'plot' arg here.
        
        step_plot = step[-1] == 1 if len(step) > 1 else False

        if method == 'MA':
            # {'MA', radius, alis_type, plot}
            radius = step[1]
            alis_type = step[2]
            absor, lambda_ = alisar(alis_type, absor, lambda_, radius)

        elif method == 'AutoScale':
            # {'AutoScale', plot}
            # Centers data and scales to unit variance
            absor, _, _ = zscore_matlab_style(absor)
            
        elif method == 'MeanCenter':
            # {'MeanCenter', plot}
            # Only centers the data
            current_mean = np.mean(absor, axis=0)
            absor = absor - current_mean
            
        elif method == 'SG':
            # {'SG', radius, alis_type, poly_order, der_order, plot}
            radius = step[1]
            alis_type = step[2] 
            poly_order = step[3]
            der_order = step[4]
            
            nd, nl = absor.shape
            absortemp = np.zeros((nd, nl))
            
            # Run sgolay_filt on each spectrum
            for idx in range(nd):
                # sgolay_filt(x, y, order, window_size, deriv)
                window_size = 2 * radius + 1

                
                res = sgolay_filt(lambda_, absor[idx, :], poly_order, window_size, der_order)
                absortemp[idx, :] = res
            
            absor = absortemp
            

            
        elif method == 'Loess':
            # {'Loess', alpha, order, plot}
            alpha = step[1]
            order = step[2]
            
            nd, nl = absor.shape
            absortemp = np.zeros_like(absor)
            
            for idx in range(nd):
                absortemp[idx, :] = loess(lambda_, absor[idx, :], alpha, order)
                
            absor = absortemp
            
        elif method == 'Deriv':
            # {'Deriv', order, plot}
            order = step[1]
            dA, d2A = diffmeu(absor, lambda_)
            if order == 1:
                absor = dA
            elif order == 2:
                absor = d2A

        elif method == 'BLCtr':
            # {'BLCtr', ini_lamb, final_lamb, Abs_value, plot}
            ini_lamb = step[1]
            final_lamb = step[2]
            target_val = step[3]
            
            # Find indices in range
            mask = (lambda_ >= ini_lamb) & (lambda_ <= final_lamb)
            
            if not np.any(mask):
                print(f"Warning: BLCtr range {ini_lamb}-{final_lamb} is empty. Skipping.")
            else:
                 # Calculate mean of each spectrum in the range (axis=1)
                 # absor shape: (samples, wavelengths)
                 current_mean = np.mean(absor[:, mask], axis=1, keepdims=True)
                 
                 # Calculate shift
                 shift = current_mean - target_val
                 
                 # Subtract shift from all wavelengths
                 absor = absor - shift
                
        elif method == 'Cut':
            # {'Cut', lb1, ub1, ..., plot}
            # args: method, lb1, ub1, [lb2, ub2]..., plot
            # Extract pairs
            # remove first and last
            ranges = step[1:-1]
            ncut = len(ranges) // 2
            
            new_absor_list = []
            new_lambda_list = []
            
            for icut in range(ncut):
                lb = ranges[icut*2]
                ub = ranges[icut*2 + 1]
                
                mask = (lambda_ >= lb) & (lambda_ <= ub)
                new_lambda_list.append(lambda_[mask])
                new_absor_list.append(absor[:, mask])
                
            if len(new_lambda_list) > 0:
                lambda_ = np.concatenate(new_lambda_list)
                absor = np.hstack(new_absor_list)

        if plot and step_plot:
            fig = plt.figure()
            plt.plot(lambda_, absor.T)
            plt.title(f"{prefix}Step {i+1}: {method}")
            plt.xlabel("Wavelength")
            plt.ylabel("Absorbance")
            # plt.show(block=False)
            if output_dir:
                fig.savefig(os.path.join(output_dir, f'{prefix}Pretreatment_{i+1}_{method}.png'))

    return absor, lambda_
