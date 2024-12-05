import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:

    def __init__(self, df):
        self.df = df.copy()
    
    def extract_time_features(self):

        df = self.df.copy()
        df['Hour'] = df['Datetime'].dt.hour
        df['Day_of_week'] = df['Datetime'].dt.dayofweek
        df['Month'] = df['Datetime'].dt.month
        
        return df
    
    def create_features(self):

        # Extract time features
        df = self.extract_time_features()
        
        # Calculate total sub-metering
        df['Total_sub_metering'] = (
            df['Sub_metering_1'] + 
            df['Sub_metering_2'] + 
            df['Sub_metering_3']
        )
        
        # Select features and target
        features = [
            'Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
            'Total_sub_metering', 'Hour', 'Day_of_week', 'Month'
        ]
        target = 'Global_active_power'
        
        X = df[features]
        y = df[target]
        
        return X, y
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):

        X, y = self.create_features()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler