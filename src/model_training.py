import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PowerConsumptionModel:

    def __init__(self, n_estimators=100, random_state=42, max_depth=10):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            max_depth=max_depth
        )
        
    def train(self, X_train, y_train):
    
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self, X):

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict(self, X_scaled, scaler=None):

        return self.model.predict(X_scaled)