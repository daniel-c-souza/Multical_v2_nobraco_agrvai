# Multical (Python Version)

A Python implementation of the Multical chemometrics toolbox, providing tools for multivariate calibration and spectral analysis. This library includes implementations of Partial Least Squares (PLS), Principal Component Regression (PCR), and Successive Projections Algorithm (SPA), along with preprocessing and analysis utilities.

## Features

- **Multivariate Calibration Models:**
  - PLS (Partial Least Squares)
  - PCR (Principal Component Regression)
  - SPA (Successive Projections Algorithm)
- **Preprocessing:**
  - Savitzky-Golay smoothing and derivatives
  - Standard Normal Variate (SNV) / MSC (implied by context)
  - Loess
- **Analysis:**
  - Lambert-Beer plots
  - PCA analysis

## Project Structure

```
.
├── src/
│   └── multical/          # Core package
│       ├── core/          # Engine and main logic
│       ├── models/        # PLS, PCR, SPA implementations
│       ├── preprocessing/ # Signal processing (Savitzky-Golay, etc.)
│       └── analysis.py    # Analysis routines
├── main_multi_ILS_new.py  # Main script for calibration/analysis
├── main_infer_ILS.py      # Script for inference
├── requirements.txt       # Python dependencies
└── ...
```

## Installation

1.  Clone the repository.
2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Calibration and Analysis

To run the main analysis and calibration pipeline, edit the configuration in `main_multi_ILS_new.py` (e.g., select method, inputs) and run:

```bash
python main_multi_ILS_new.py
```

### Inference

For running inference tasks:

```bash
python main_infer_ILS.py
```

## Data Format

The scripts expect input data files (e.g., `.txt` files for absorbance and concentration) as defined in the loading sections of the main scripts. Ensure your data follows the expected format (typically numeric matrices or vectors).
