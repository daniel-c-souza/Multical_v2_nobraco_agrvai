
try:
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from scipy.stats import f as f_dist
    from src.multical.core.engine import MulticalEngine
    from src.multical.models.pls import PLS
    from src.multical.utils import zscore_matlab_style
    from src.multical.preprocessing.pipeline import apply_pretreatment
    from src.multical.analysis import func_analysis
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
