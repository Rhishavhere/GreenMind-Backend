import matplotlib.pyplot as plt
import seaborn as sns

class EnergyVisualizer:
    @staticmethod
    def plot_prediction_accuracy(y_test, y_pred):

        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Global Active Power (kW)')
        plt.ylabel('Predicted Global Active Power (kW)')
        plt.title('Prediction vs Actual')
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance):

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
        plt.figure(figsize=(12, 6))
        hourly_consumption.index = [f"{h:02d}:{m:02d}" for h, m in hourly_consumption.index]
        hourly_consumption.plot(kind='line', marker='o')
        plt.title('Average Energy Consumption by Minute')
        plt.xlabel('Time of the Day (Hour:Minute)')
        plt.ylabel('Average Global Active Power (kW)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()