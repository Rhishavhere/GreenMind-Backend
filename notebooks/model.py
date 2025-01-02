import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
def load_power_consumption_data(filepath):
    """
    Load power consumption data from a CSV file.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        # Read CSV and parse datetime
        df = pd.read_csv(filepath, parse_dates=[['Date', 'Time']])
        
        # Rename combined column
        df.rename(columns={'Date_Time': 'Datetime'}, inplace=True)
        
        # Replace '?' with NaN and convert to numeric
        numeric_columns = [
            'Global_active_power', 'Global_reactive_power', 
            'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess the data
def preprocess_data(df):
    """
    Preprocess the power consumption dataset.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    tuple: Processed features (X) and target variable (y)
    """
    # Extract time-based features
    df['Hour'] = df['Datetime'].dt.hour
    df['Day_of_week'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    
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

# Train the model
def train_power_prediction_model(X, y):
    """
    Train a Random Forest Regressor for power consumption prediction.
    
    Parameters:
    X (pandas.DataFrame): Feature dataset
    y (pandas.Series): Target variable
    
    Returns:
    tuple: Trained model, scaler, test data, predictions, and evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_test, y_test, y_pred, {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }, feature_importance

# Visualization function
def visualize_predictions(y_test, y_pred, feature_importance):
    """
    Create visualizations of model performance and feature importance.
    
    Parameters:
    y_test (array-like): Actual test values
    y_pred (array-like): Predicted values
    feature_importance (pandas.DataFrame): Feature importance data
    """
    # Create a figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Prediction vs Actual Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Global Active Power (kW)')
    plt.ylabel('Predicted Global Active Power (kW)')
    plt.title('Prediction vs Actual')
    
    # Feature Importance Plot
    plt.subplot(1, 2, 2)
    feature_importance.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
    plt.title('Feature Importance for Power Prediction')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Main execution function
def main(filepath):
    """
    Main function to load data, train model, and visualize results.
    
    Parameters:
    filepath (str): Path to the power consumption CSV
    """
    # Load data
    df = load_power_consumption_data(filepath)
    if df is None:
        return
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Train model
    model, scaler, X_test, y_test, y_pred, metrics, feature_importance = train_power_prediction_model(X, y)
    
    # Print metrics
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Visualize results
    visualize_predictions(y_test, y_pred, feature_importance)
    
    return model, scaler

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    filepath = 'household_power_consumption.csv'
    model, scaler = main(filepath)

# Optional: Function for making predictions on new data
def predict_power_consumption(model, scaler, new_data):
    """
    Make predictions on new data.
    
    Parameters:
    model: Trained Random Forest model
    scaler: Feature scaler
    new_data (pandas.DataFrame): New data for prediction
    
    Returns:
    numpy.ndarray: Predicted power consumption
    """
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    
    return predictions