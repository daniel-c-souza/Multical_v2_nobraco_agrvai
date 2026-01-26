import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from src.multical.core.engine import MulticalEngine
# from src.multical.analysis import func_analysis # Optional if we want analysis buttons

class MulticalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multical Interface")
        self.root.geometry("900x700")
        
        # --- Variables ---
        self.file_pairs = [] # List of tuples (conc_path, abs_path)
        
        self.var_method = tk.IntVar(value=1) # 1=PLS, 2=SPA, 3=PCR
        self.var_kmax = tk.IntVar(value=20)
        self.var_nc = tk.IntVar(value=3)
        self.var_cname = tk.StringVar(value="cb, gl, xy")
        self.var_unid = tk.StringVar(value="g/L")
        self.var_out_dir = tk.StringVar(value="results_gui")
        self.var_outlier = tk.IntVar(value=0)
        self.var_ftest = tk.BooleanVar(value=False)
        
        # Validation
        self.var_val_type = tk.StringVar(value="kfold") # kfold, holdout
        self.var_val_param = tk.DoubleVar(value=10) # kpart or fraction
        self.var_cv_type = tk.StringVar(value="venetian") # venetian, random, consecutive

        # Analysis
        self.var_anal_lb = tk.BooleanVar(value=True)
        self.var_anal_pca = tk.BooleanVar(value=True)
        
        # Session
        self.var_autosave = tk.BooleanVar(value=True)
        self.session_file = "multical_session.json"
        
        # =========================================================================
        #                                COLOR SETTINGS
        # =========================================================================
        # You can easily edit these default colors here:
        self.current_bg = "#eef2f7"
        self.current_fg = "#1e293b"

        self.entry_bg   = "#ffffff"
        self.entry_fg   = "#1e293b"

        self.btn_bg     = "#dbe2ec"
        self.btn_fg     = "#1e293b"

        self.select_bg  = "#0ea5e9"   # Azul 
        self.select_fg  = "#ffffff"




        # =========================================================================

        # Styles and Colors
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam' generally supports color definitions better

        # --- Layout ---
        self._create_widgets()
        
        # Apply initial theme
        self.apply_theme()
        
        # Load Session
        self.load_session()
        
        # Handle Exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.save_session()
        self.root.destroy()

    def load_session(self):
        if not os.path.exists(self.session_file):
            return
            
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)
                
            # Restore Autosave Preference first
            if 'autosave' in data:
                self.var_autosave.set(data['autosave'])
                
            if not self.var_autosave.get():
                return 

            # Restore File Pairs
            if 'file_pairs' in data:
                saved_pairs = data['file_pairs']
                for c_file, a_file in saved_pairs:
                    if os.path.exists(c_file) and os.path.exists(a_file):
                        self.file_pairs.append((c_file, a_file))
                        self.lst_files.insert(tk.END, f"{os.path.basename(c_file)}  |  {os.path.basename(a_file)}")
            
            # Restore Other Settings if likely useful
            if 'out_dir' in data: self.var_out_dir.set(data['out_dir'])
            if 'method' in data: self.var_method.set(data['method'])
            if 'kmax' in data: self.var_kmax.set(data['kmax'])
            if 'nc' in data: self.var_nc.set(data['nc'])
            if 'cname' in data: self.var_cname.set(data['cname'])
            
        except Exception as e:
            print(f"Error loading session: {e}")

    def save_session(self):
        if not self.var_autosave.get():
            return
            
        data = {
            'autosave': self.var_autosave.get(),
            'file_pairs': self.file_pairs,
            'out_dir': self.var_out_dir.get(),
            'method': self.var_method.get(),
            'kmax': self.var_kmax.get(),
            'nc': self.var_nc.get(),
            'cname': self.var_cname.get()
        }
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving session: {e}")

    def _create_widgets(self):
        # Menu Bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        app_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Appearance", menu=app_menu)
        app_menu.add_command(label="Choose Background Color", command=self.choose_bg_color)
        app_menu.add_command(label="Choose Text Color", command=self.choose_fg_color)
        app_menu.add_separator()
        app_menu.add_command(label="Apply to Output Plots (Matplotlib)", command=self.apply_to_plots_toggle) # Placeholder/Feature

        # Top Frame: Configuration
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Method
        ttk.Label(config_frame, text="Method:").grid(row=0, column=0, sticky="w")
        methods = [("PLS", 1), ("SPA", 2), ("PCR", 3)]
        m_frame = ttk.Frame(config_frame)
        m_frame.grid(row=0, column=1, sticky="w")
        for text, val in methods:
            ttk.Radiobutton(m_frame, text=text, variable=self.var_method, value=val).pack(side=tk.LEFT, padx=2)
            
        # Kmax, NC
        ttk.Label(config_frame, text="Max Latent Vars (kmax):").grid(row=1, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_kmax, width=5).grid(row=1, column=1, sticky="w")
        
        ttk.Label(config_frame, text="Num Components (nc):").grid(row=2, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_nc, width=5).grid(row=2, column=1, sticky="w")
        
        # Names
        ttk.Label(config_frame, text="Constituents (comma sep):").grid(row=3, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_cname).grid(row=3, column=1, sticky="ew")
        
        ttk.Label(config_frame, text="Units:").grid(row=4, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_unid, width=10).grid(row=4, column=1, sticky="w")
        
        # Validation
        ttk.Label(config_frame, text="Validation Type:").grid(row=5, column=0, sticky="w")
        val_cb = ttk.Combobox(config_frame, textvariable=self.var_val_type, values=["kfold", "holdout"], state="readonly")
        val_cb.grid(row=5, column=1, sticky="w")
        
        ttk.Label(config_frame, text="Val Param (Folds/%):").grid(row=6, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_val_param, width=10).grid(row=6, column=1, sticky="w")
        
        ttk.Label(config_frame, text="CV Type (if K-Fold):").grid(row=7, column=0, sticky="w")
        cv_cb = ttk.Combobox(config_frame, textvariable=self.var_cv_type, values=["venetian", "random", "consecutive"], state="readonly")
        cv_cb.grid(row=7, column=1, sticky="w")

        # Options
        ttk.Checkbutton(config_frame, text="Remove Outliers (Std t-test)", variable=self.var_outlier).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(config_frame, text="Use Osten F-Test", variable=self.var_ftest).grid(row=9, column=0, columnspan=2, sticky="w")
        
        # Analysis Options
        ttk.Label(config_frame, text="Run Analysis:").grid(row=10, column=0, sticky="w", pady=(5,0))
        anal_frame = ttk.Frame(config_frame)
        anal_frame.grid(row=10, column=1, sticky="w")
        ttk.Checkbutton(anal_frame, text="Lambert-Beer", variable=self.var_anal_lb).pack(side=tk.LEFT)
        ttk.Checkbutton(anal_frame, text="PCA", variable=self.var_anal_pca).pack(side=tk.LEFT)

        # Pretreatment
        ttk.Label(config_frame, text="Pretreatment (Python List syntax):").grid(row=11, column=0, sticky="w", pady=(10, 0))
        self.txt_pretreat = scrolledtext.ScrolledText(config_frame, height=8, width=40)
        self.txt_pretreat.grid(row=12, column=0, columnspan=2, sticky="ew")
        
        default_pretreat = "[\n  ['Cut', 5500, 8500, 1],\n  ['SG', 7, 1, 2, 1, 1]\n]"
        self.txt_pretreat.insert(tk.END, default_pretreat)
        
        # Directory
        ttk.Label(config_frame, text="Output Directory:").grid(row=13, column=0, sticky="w")
        ttk.Entry(config_frame, textvariable=self.var_out_dir).grid(row=13, column=1, sticky="ew")
        
        # Session Options
        ttk.Checkbutton(config_frame, text="Restore files on startup", variable=self.var_autosave).grid(row=14, column=0, columnspan=2, sticky="w", pady=5)

        # --- Right Frame: Data & Actions ---
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=10)
        data_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lst_files = tk.Listbox(data_frame, width=50, height=15)
        self.lst_files.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Add Data Pair...", command=self.add_data_pair).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear List", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close All Plots", command=self.close_plots).pack(side=tk.LEFT, padx=5)
        
        # Action Button
        ttk.Button(data_frame, text="RUN ANALYSIS", command=self.run_analysis).pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    def close_plots(self):
        plt.close('all')

    def add_data_pair(self):
        # Simple implementation: Ask for Conc file, then Abs file
        conc_file = filedialog.askopenfilename(title="Select Concentration File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not conc_file: return
        
        abs_file = filedialog.askopenfilename(title="Select Absorbance File for " + os.path.basename(conc_file), filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not abs_file: return
        
        self.file_pairs.append((conc_file, abs_file))
        self.lst_files.insert(tk.END, f"{os.path.basename(conc_file)}  |  {os.path.basename(abs_file)}")

    def clear_data(self):
        self.file_pairs = []
        self.lst_files.delete(0, tk.END)

    def run_analysis(self):
        if not self.file_pairs:
            messagebox.showerror("Error", "No data files selected.")
            return

        # 1. Load Data
        try:
             x0, absor0 = self.load_data(self.file_pairs)
        except Exception as e:
             messagebox.showerror("Loading Error", f"Failed to load data:\n{e}")
             return
             
        # 2. Parse Parameters
        try:
            Selecao = self.var_method.get()
            kmax = self.var_kmax.get()
            nc = self.var_nc.get()
            cname_str = self.var_cname.get()
            cname = [s.strip() for s in cname_str.split(",")]
            unid = self.var_unid.get()
            out_dir = self.var_out_dir.get()
            outlier = self.var_outlier.get()
            use_ftest = self.var_ftest.get()
            
            # Validation
            val_type = self.var_val_type.get()
            val_param = self.var_val_param.get()
            cv_type_sel = self.var_cv_type.get()
            
            if val_type == "kfold":
                # ['kfold', kpart, cv_type]
                OptimModel = ['kfold', int(val_param), cv_type_sel] 
            else:
                # ['Val', frac_val]
                OptimModel = ['Val', float(val_param)] # Assuming param is fraction like 0.2
            
            # Analysis List
            analysis_list = []
            if self.var_anal_lb.get():
                analysis_list.append(['LB'])
            if self.var_anal_pca.get():
                analysis_list.append(['PCA'])
            
            if len(analysis_list) == 0:
                analysis_list = None
            
            # Pretreatment
            pretreat_str = self.txt_pretreat.get("1.0", tk.END).strip()
            pretreat = eval(pretreat_str) # Be careful with eval
            
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Invalid parameters:\n{e}")
            return
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 3. Execution
        try:
            engine = MulticalEngine()
            
            # Fixed params for GUI simplification for now
            optkini = 2 
            lini = 0 
            frac_test = 0.0
            dadosteste = []
            
            print("Starting Analysis...")
            self.root.update() # Update UI
            
            RMSECV, _, RMSEcal, _, _, _ = engine.run(
                Selecao, optkini, lini, kmax, nc, cname, unid, x0, absor0, 
                frac_test, dadosteste, OptimModel, pretreat, 
                analysis_list=analysis_list, output_dir=out_dir, outlier=outlier, use_ftest=use_ftest
            )
            
            messagebox.showinfo("Success", "Analysis Complete! Check results folder.")
            
        except Exception as e:
            messagebox.showerror("Execution Error", f"An error occurred during execution:\n{e}")
            print(e)

    def load_data(self, data_files):
        x_list = []
        absor_list = []
        time_list_conc = [] 
        time_list_spec = [] 
        wavelengths = None
        
        for x_f, abs_f in data_files:
            if os.path.exists(x_f) and os.path.exists(abs_f):
                print(f"Loading: {x_f} / {abs_f}")
                xi = np.loadtxt(x_f)
                
                # Load absorbance file with header handling
                with open(abs_f, 'r') as f_node:
                    header_parts = f_node.readline().strip().split()
                
                # Extract wavelengths (skip "Time")
                wl_curr = np.array([float(x) for x in header_parts[1:]])
                
                # Load data (skip header row)
                absi = np.loadtxt(abs_f, skiprows=1)
                
                # Check for Time column in Concentration File
                if xi.ndim == 2 and xi.shape[1] > 1:
                    xi = xi[:, 1:] 
                else:
                    if xi.ndim == 1:
                        xi = xi.reshape(-1, 1)
                
                data_curr = absi[:, 1:] # Skip first col (time)
                
                if wavelengths is None:
                    wavelengths = wl_curr
                else:
                    if not np.allclose(wavelengths, wl_curr, atol=1e-1):
                         print(f"Warning: Wavelengths in {abs_f} differ from previous.")
                
                x_list.append(xi)
                absor_list.append(data_curr)
            else:
                raise FileNotFoundError(f"File not found: {x_f} or {abs_f}")

        if x_list:
            x0 = np.vstack(x_list)
            absor_data = np.vstack(absor_list)
            # Combine: Row 0 is wavelengths
            absor0 = np.vstack([wavelengths, absor_data])
            return x0, absor0
        else:
            raise ValueError("No data loaded")

    def choose_bg_color(self):
        color = colorchooser.askcolor(title="Choose Background Color")[1]
        if color:
            self.current_bg = color
            self.apply_theme()

    def choose_fg_color(self):
        color = colorchooser.askcolor(title="Choose Text Color")[1]
        if color:
            self.current_fg = color
            self.apply_theme()
            
    def apply_to_plots_toggle(self):
        # We can set matplotlib params globally if desired
        # For now just a message since dynamic plot updating needs more logic in engine
        pass

    def apply_theme(self):
        bg = self.current_bg
        fg = self.current_fg
        e_bg = self.entry_bg
        e_fg = self.entry_fg
        b_bg = self.btn_bg
        b_fg = self.btn_fg
        s_bg = self.select_bg
        s_fg = self.select_fg
        
        # Configure Root
        self.root.config(bg=bg)
        
        # Configure TTK Style
        # '.' styles essentially all tile widgets
        self.style.configure(".", background=bg, foreground=fg)
        
        # Some widgets need specific configs in 'clam' or other themes
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabelframe", background=bg, foreground=fg)
        self.style.configure("TLabelframe.Label", background=bg, foreground=fg)
        self.style.configure("TCheckbutton", background=bg, foreground=fg)
        self.style.configure("TRadiobutton", background=bg, foreground=fg)
        
        # Buttons
        self.style.configure("TButton", background=b_bg, foreground=b_fg)
        
        # Configure Entry/Combobox fields (with selection colors)
        self.style.configure("TEntry", fieldbackground=e_bg, foreground=e_fg, 
                             selectbackground=s_bg, selectforeground=s_fg)
        self.style.configure("TCombobox", fieldbackground=e_bg, foreground=e_fg, background=bg,
                             selectbackground=s_bg, selectforeground=s_fg)
        
        # Configure Non-TTK Widgets directly
        if hasattr(self, 'lst_files'):
            self.lst_files.config(bg=e_bg, fg=e_fg, selectbackground=s_bg, selectforeground=s_fg) 
            
        if hasattr(self, 'txt_pretreat'):
            self.txt_pretreat.config(bg=e_bg, fg=e_fg, insertbackground=fg, 
                                     selectbackground=s_bg, selectforeground=s_fg)

if __name__ == "__main__":
    root = tk.Tk()
    app = MulticalGUI(root)
    root.mainloop()
