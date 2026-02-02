import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from copy import deepcopy
import joblib
import os

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', dropout_prob=0.0):
        super(MLP, self).__init__()
        
        # Select Activation
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'tanh':
            self.act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'silu':
            self.act_fn = nn.SiLU()
        elif activation == 'elu':
            self.act_fn = nn.ELU()
        else:
            raise ValueError(f"Activation {activation} not supported.")
            
        self.dropout = nn.Dropout(dropout_prob)
        
        layers = []
        # Input Layer -> First Hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.act_fn)
        layers.append(self.dropout)
        
        # Additional Hidden Layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act_fn)
            layers.append(self.dropout)
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ANNModel:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.input_dim = None
        self.output_dim = None
        self.hidden_units = None
        self.n_layers = None
        self.activation = None

    def fit_predict(self, X_train, Y_train, X_test, 
                   hidden_units=50, n_layers=1, activation='relu',
                   learning_rate=0.001, epochs=200, 
                   batch_size=32, device=None, early_stopping=False, patience=20):
        """
        Fits PyTorch ANN and predicts on X_test.
        Handles multi-output automatically (output_dim = nc).
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.activation = activation

        # 1. Scale Data
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        
        if Y_train.ndim == 1: Y_train = Y_train.reshape(-1, 1)
        Y_train_scaled = self.scaler_y.fit_transform(Y_train)
        
        X_test_scaled = self.scaler_x.transform(X_test)
        
        # Handle Validation Split for Early Stopping
        X_tr_final = X_train_scaled
        Y_tr_final = Y_train_scaled
        X_val_final = None
        Y_val_final = None

        if early_stopping:
            # 10% validation split
            n_samples_total = X_train_scaled.shape[0]
            n_val = max(int(n_samples_total * 0.1), 1)
            n_train = n_samples_total - n_val
            
            # Use deterministic split for monitoring
            perm = np.random.RandomState(42).permutation(n_samples_total)
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
            
            X_tr_final = X_train_scaled[train_idx]
            Y_tr_final = Y_train_scaled[train_idx]
            X_val_final = X_train_scaled[val_idx]
            Y_val_final = Y_train_scaled[val_idx]

        # Convert to Tensors
        X_tr = torch.tensor(X_tr_final, dtype=torch.float32).to(device)
        Y_tr = torch.tensor(Y_tr_final, dtype=torch.float32).to(device)
        X_ts = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        if early_stopping and X_val_final is not None:
             X_v = torch.tensor(X_val_final, dtype=torch.float32).to(device)
             Y_v = torch.tensor(Y_val_final, dtype=torch.float32).to(device)
        
        # Dimensions
        self.input_dim = X_tr.shape[1]
        self.output_dim = Y_tr.shape[1]
        
        # 2. Initialize Model
        # Using a deterministic seed for reproducibility within the fold
        torch.manual_seed(42)
        self.model = MLP(self.input_dim, hidden_dim=int(hidden_units), output_dim=self.output_dim,
                   n_layers=n_layers, activation=activation).to(device)
        
        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 3. Training Loop
        self.model.train()
        n_samples = X_tr.shape[0]
        
        # Early Stopping Variables
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Simple Mini-batch training
        for epoch in range(epochs):
            # Shuffle indices
            permutation = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_tr[indices], Y_tr[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            # Early Stopping Check
            if early_stopping and X_val_final is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_loss = criterion(val_out, Y_v).item()
                self.model.train()
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    # Restore best model
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
        
        # Restore best model if loop finished without breaking but early stopping was on
        if early_stopping and best_model_state is not None:
             self.model.load_state_dict(best_model_state)

        # 4. Predict
        self.model.eval()
        with torch.no_grad():
            Y_pred_scaled_t = self.model(X_ts)
            Y_pred_scaled = Y_pred_scaled_t.cpu().numpy()
            
        # 5. Inverse Scale
        Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
        
        return Y_pred

    def save_model(self, path):
        """Saves model weights and scalers."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save PyTorch Model
        torch.save(self.model.state_dict(), path + '.pth')
        
        # Save Metadata & Scalers
        metadata = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_units': self.hidden_units,
            'n_layers': self.n_layers,
            'activation': self.activation,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y
        }
        joblib.dump(metadata, path + '.meta')
        print(f"Model saved to {path}.pth and {path}.meta")

    def load_model(self, path, device=None):
        """Loads model weights and scalers."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if not os.path.exists(path + '.pth') or not os.path.exists(path + '.meta'):
            raise FileNotFoundError(f"Model files not found at {path}")
            
        # Load Metadata
        metadata = joblib.load(path + '.meta')
        self.input_dim = metadata['input_dim']
        self.output_dim = metadata['output_dim']
        self.hidden_units = metadata['hidden_units']
        self.n_layers = metadata['n_layers']
        self.activation = metadata['activation']
        self.scaler_x = metadata['scaler_x']
        self.scaler_y = metadata['scaler_y']
        
        # Re-init Model
        self.model = MLP(self.input_dim, hidden_dim=int(self.hidden_units), output_dim=self.output_dim,
                         n_layers=self.n_layers, activation=self.activation).to(device)
        
        # Load Weights
        self.model.load_state_dict(torch.load(path + '.pth', map_location=device, weights_only=True))
        self.model.eval()
        print(f"Model loaded from {path}")

    def predict(self, X_new, device=None):
        """Predicts on new data using loaded model."""
        if self.model is None:
            raise ValueError("Model not loaded/trained.")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.to(device)
        self.model.eval()
        
        # Scale Input
        X_scaled = self.scaler_x.transform(X_new)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            Y_pred_scaled_t = self.model(X_tensor)
            Y_pred_scaled = Y_pred_scaled_t.cpu().numpy()
            
        # Inverse Scale Output
        Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
        return Y_pred
