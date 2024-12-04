import matplotlib.pyplot as plt
import seaborn as sns

class EnergyVisualizer:
    """
    Class for creating visualizations of energy consumption data
    """
    @staticmethod
    def plot_prediction_accuracy(y_test, y_pred):
        """
        Create scatter plot of prediction accuracy
        
        Args:
            y_test (array-like): Actual test values
            y_pred (array-like): Predicted values
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Global Active Power (kW)')
        plt.ylabel('Predicted Global Active Power (kW)')
        plt.title('Prediction vs Actual')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance):
        """
        Create bar plot of feature importances
        
        Args:
            feature_importance (pandas.DataFrame): Feature importance data
        """
        plt.figure(figsize=(10, 5))
        feature_importance.plot(x='feature', y='importance', kind='bar')
        plt.title('Feature Importance for Power Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_hourly_consumption(hourly_consumption):
        """
        Create line plot of hourly energy consumption
        
        Args:
            hourly_consumption (pandas.Series): Average consumption by hour
        """
        plt.figure(figsize=(12, 5))
        hourly_consumption.plot(kind='line', marker='o')
        plt.title('Average Energy Consumption by Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Global Active Power (kW)')
        plt.xticks(range(24))
        plt.grid(True)
        plt.show()