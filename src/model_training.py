import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PowerConsumptionModel:
    """
    Class for training and evaluating power consumption prediction model
    """
    def __init__(self, n_estimators=100, random_state=42, max_depth=10):
        """
        Initialize Random Forest Regressor
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random seed for reproducibility
            max_depth (int): Maximum depth of trees
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            max_depth=max_depth
        )
        
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Scaled training features
            y_train (numpy.ndarray): Training target variable
        """
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Scaled test features
            y_test (numpy.ndarray): Test target variable
        
        Returns:
            dict: Performance metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self, X):
        """
        Get feature importances
        
        Args:
            X (pandas.DataFrame): Feature names
        
        Returns:
            pandas.DataFrame: Feature importance ranking
        """
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, X_scaled, scaler=None):
        """
        Make predictions on new data
        
        Args:
            X_scaled (numpy.ndarray): Scaled input features
            scaler (StandardScaler, optional): Feature scaler
        
        Returns:
            numpy.ndarray: Predictions
        """
        return self.model.predict(X_scaled)