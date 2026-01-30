
import pandas as pd
import numpy as np
import os

def process_file(filename_in, filename_out):
    print(f"Processing {filename_in} -> {filename_out}")
    if not os.path.exists(filename_in):
        print(f"Error: {filename_in} not found.")
        return

    # 1. Read CSV. 
    try:
        df = pd.read_csv(filename_in, sep=';', header=0)
    except Exception as e:
        print(f"Failed to read csv with ';': {e}")
        return

    # Structure:
    # Col 0: "Wavenumber" (values are the wavelengths)
    # Col 1..N: "0.0 min", "3.0 min", ... (values are spectra)
    
    # We want:
    # Row 0: "Time", Wav1, Wav2, ...
    # Row 1..N: TimeVal, Abs1, Abs2, ...

    # Extract Wavelengths
    wavenumbers = df["Wavenumber"].values # These will become columns
    
    # Extract Data (drop Wavenumber col)
    data = df.drop(columns=["Wavenumber"])
    
    # The columns of 'data' are the Time points (strings "0.0 min", etc.)
    # The rows of 'data' correspond to 'wavenumbers'
    
    # We need to transpose: Rows=Time, Cols=Wavenumber
    data_T = data.T # Now index is "0.0 min"..., columns are 0..M (indices of wavenumbers)
    
    # Set the column names to be the wavenumbers
    data_T.columns = wavenumbers
    
    # Reset index to get the Time columns into a real column
    data_T = data_T.reset_index()
    data_T.rename(columns={"index": "Time"}, inplace=True)
    
    # Parse proper numerical Time values.
    def parse_time(s):
        try:
            return float(s.replace(' min', '').strip())
        except:
            return -1 # Fallback
            
    data_T["Time"] = data_T["Time"].apply(parse_time)
    
    # Sort columns by Wavelength (Low -> High) to match exp4_nonda.txt format
    # 'Time' column should stay first.
    
    # Get numeric columns (wavenumbers)
    wav_cols = [c for c in data_T.columns if c != "Time"]
    
    # Create valid mapping
    col_map = {c: float(c) for c in wav_cols}
    # Sort keys based on float values
    sorted_keys = sorted(col_map.keys(), key=lambda x: col_map[x])
    
    final_cols = ["Time"] + sorted_keys
    final_df = data_T[final_cols]
    
    # Save as tab-separated
    final_df.to_csv(filename_out, sep='\t', index=False, float_format='%.6f')
    print(f"Done. Saved to {filename_out}")

files = [
    ("exp_04_inf.csv", "exp_04_inf.txt"),
    ("exp_05_inf.csv", "exp_05_inf.txt"),
    ("exp_06_inf.csv", "exp_06_inf.txt")
]

if __name__ == "__main__":
    for fin, fout in files:
        process_file(fin, fout)
