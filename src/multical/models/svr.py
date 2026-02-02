import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SVRModel:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None

    def fit_predict(self, X_train, Y_train, X_test, C=1.0, epsilon=0.1, gamma='scale', kernel='rbf'):
        """
        Fits SVR and predicts on X_test.
        Handles multi-output using MultiOutputRegressor.
        """
        # 1. Scale Data
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
            
        Y_train_scaled = self.scaler_y.fit_transform(Y_train)
        
        # 2. Define Model
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model = MultiOutputRegressor(svr)
        
        # 3. Fit
        self.model.fit(X_train_scaled, Y_train_scaled)
        
        # 4. Predict
        if X_test is not None:
             X_test_scaled = self.scaler_x.transform(X_test)
             Y_pred_scaled = self.model.predict(X_test_scaled)
             Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
             return Y_pred
        return None
        
    def predict(self, X):
        """
        Predicts using trained model.
        """
        if self.model is None or self.scaler_x is None:
            raise ValueError("Model not trained.")
            
        X_scaled = self.scaler_x.transform(X)
        Y_pred_scaled = self.model.predict(X_scaled)
        Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
        return Y_pred
        
    def save_model(self, path):
        """
        Saves the model and scalers using joblib.
        """
        if self.model is None:
            raise ValueError("No model to save.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # We save everything in a dict wrapper to keep single file
        data = {
            'model': self.model,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y
        }
        if hasattr(self, 'custom_y_scale'):
            data['custom_y_scale'] = self.custom_y_scale
            
        if not path.endswith('.pkl'):
            path += '.pkl'
        joblib.dump(data, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Loads the model.
        """
        if not path.endswith('.pkl'):
            path += '.pkl'
            
        data = joblib.load(path)
        self.model = data['model']
        self.scaler_x = data['scaler_x']
        self.scaler_y = data['scaler_y']
        if 'custom_y_scale' in data:
            self.custom_y_scale = data['custom_y_scale']

