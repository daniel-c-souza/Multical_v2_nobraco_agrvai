import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def wavenumber_to_wavelength(wavenumber):
    """
    Converts wavenumber (cm-1) to wavelength (nm).
    lambda = 10^7 / nu
    """
    # Avoid division by zero if present
    return 10_000_000 / wavenumber

def process_nonda_files():
    # Find all exp*_nonda.txt files
    files = glob.glob('exp*_nonda.txt')
    
    for file in files:
        print(f"Processing {file}...")
        try:
            # Read 'wide' format: Time, 1200, 1201, ...
            # Using sep='\s+' to handle tabs or spaces
            df = pd.read_csv(file, sep='\s+')
            
            # Identify spectral columns (numbers)
            cols = df.columns
            # Filter for numeric columns (wavenumbers)
            wn_cols = [c for c in cols if c.replace('.', '').isdigit()]
            meta_cols = [c for c in cols if c not in wn_cols]
            
            if not wn_cols:
                print(f"Warning: No wavenumber columns found in {file}. Skipping.")
                continue
                
            wavenumbers = np.array([float(x) for x in wn_cols])
            wavelengths = wavenumber_to_wavelength(wavenumbers)
            
            # Create new column names: keep meta_cols, replace wn_cols with wavelengths
            # Note: Wavelengths will likely be non-integers, string format them
            new_wn_cols = [f"{wl:.2f}" for wl in wavelengths]
            
            # Create a copy for saving (renaming columns)
            df_nm = df.copy()
            # Map old names to new names
            file_rename_map = dict(zip(wn_cols, new_wn_cols))
            df_nm = df_nm.rename(columns=file_rename_map)
            
            # Save converted file
            base_name = os.path.splitext(file)[0]
            output_file = f"{base_name}_nm.csv"
            df_nm.to_csv(output_file, index=False)
            print(f"Saved {output_file}")
            
            # Plotting
            plt.figure(figsize=(10, 6))
            # X-axis: Wavelengths (nm)
            # Y-axis: Intensities (rows of the dataframe)
            
            # Plot all rows (spectra at different times)
            # We iterate over rows
            for idx, row in df.iterrows():
                # Get intensities for spectral columns
                intensities = row[wn_cols].values.astype(float)
                
                # Label with Time if available
                label = f"Time {row['Time']}" if 'Time' in df.columns else f"Row {idx}"
                # Plot only a few lines if too many, or plot all with thin lines
                if len(df) > 50: # If many time points, plot without individual labels or subset
                     plt.plot(wavelengths, intensities, alpha=0.5, color='blue', linewidth=0.5)
                else:
                     plt.plot(wavelengths, intensities, label=label)

            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Absorbance / Intensity')
            plt.title(f'Spectra converted to Wavelength: {file}')
            if len(df) <= 20: 
                plt.legend()
            elif 'Time' in df.columns:
                # Add a text box or colorbar explanation if needed, or just simple plot
                pass
                
            plt.gca().invert_xaxis() # Convention for nm usually? Or low-to-high? 
            # NIR is usually plotted low-to-high (800-2500nm). 
            # Wavenumber was 1200->10000 (decreasing wavelength).
            # 10^7/1200 = 8333 nm. 10^7/10000 = 1000 nm.
            # plt.plot uses array order. If we want standard X axis (low left, high right), it's fine.
            
            plot_file = f"{base_name}_nm_plot.png"
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved plot {plot_file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

def process_inf_files():
    # Find exp_*_inf.csv files (ignoring empty .txt)
    # Pattern: exp_*_inf.csv
    files = glob.glob('exp_*_inf.csv')
    
    for file in files:
        print(f"Processing {file}...")
        try:
            # Read 'tall' format: Wavenumber, 0.0 min, 3.0 min...
            # These were semicolon separated
            df = pd.read_csv(file, sep=';')
            
            if 'Wavenumber' not in df.columns:
                # Try reading header differently or different sep?
                # The file read previously showed "Wavenumber;0.0 min..."
                if df.shape[1] < 2:
                     # Maybe it read as one column?
                     print(f"Structure check failed for {file}. Columns: {df.columns}")
                     continue
            
            # Get wavenumbers column
            # Check if index 0 is wavenumber or specific column 'Wavenumber'
            if 'Wavenumber' in df.columns:
                wavenumbers = df['Wavenumber'].astype(float)
                # Drop Wavenumber to get data columns
                data_df = df.drop(columns=['Wavenumber'])
            else:
                # Fallback: Assume first column
                wavenumbers = df.iloc[:, 0].astype(float)
                data_df = df.iloc[:, 1:]
            
            wavelengths = wavenumber_to_wavelength(wavenumbers)
            
            # Create new dataframe with Wavelength
            df_nm = data_df.copy()
            df_nm.insert(0, 'Wavelength', wavelengths)
            
            # Save
            base_name = os.path.splitext(file)[0]
            output_file = f"{base_name}_nm.csv"
            df_nm.to_csv(output_file, index=False)
            print(f"Saved {output_file}")
            
            # Plotting
            plt.figure(figsize=(10, 6))
            
            # Iterate columns (Time points)
            for col in data_df.columns:
                intensities = data_df[col].values
                plt.plot(wavelengths, intensities, label=col, linewidth=0.8)
                
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Absorbance / Intensity')
            plt.title(f'Inference Spectra converted to Wavelength: {file}')
            
            # Legend might be big if many time points
            if len(data_df.columns) <= 10:
                plt.legend()
            else:
                # Maybe no legend or outside
                pass
                
            plt.savefig(f"{base_name}_nm_plot.png")
            plt.close()
            print(f"Saved plot {base_name}_nm_plot.png")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    process_nonda_files()
    process_inf_files()
