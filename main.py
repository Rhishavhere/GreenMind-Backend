import logging
import os
from src.__init__ import PROJECT_CONFIG

# Import project modules
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_training import PowerConsumptionModel
from src.energy_analysis import EnergyAnalyzer
from src.visualization import EnergyVisualizer

def main():
    """
    Main workflow for power consumption analysis and modeling
    """
    logger = logging.getLogger(__name__)

    try:
        # Construct full path to the dataset
        DATA_FILE_PATH = os.path.join(PROJECT_CONFIG['data_dir'], 'household_power_consumption.csv')
        
        logger.info(f"Starting analysis for project: {PROJECT_CONFIG['name']}")
        logger.info(f"Using dataset: {DATA_FILE_PATH}")

        # 1. Data Loading
        logger.info("Starting data loading process...")
        data_loader = DataLoader(DATA_FILE_PATH)
        df = data_loader.load_data()

        # 2. Data Preprocessing
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(df)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocessor.prepare_train_test_split()

        # 3. Model Training
        logger.info("Training power consumption prediction model...")
        model = PowerConsumptionModel()
        model.train(X_train_scaled, y_train)

        # 4. Model Evaluation
        logger.info("Evaluating model performance...")
        metrics = model.evaluate(X_test_scaled, y_test)
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")

        # 5. Feature Importance Analysis
        logger.info("Analyzing feature importances...")
        X_before_scaling = preprocessor.create_features()[0]
        feature_importance = model.get_feature_importance(X_before_scaling)
        
        # 6. Energy Analysis
        logger.info("Performing energy consumption analysis...")
        energy_analyzer = EnergyAnalyzer(df)
        energy_analyzer.generate_sustainability_report()

        # 7. Visualizations
        logger.info("Creating visualizations...")
        # Prepare output directory for plots
        output_dir = PROJECT_CONFIG['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Prediction Accuracy Plot
        y_pred = model.predict(X_test_scaled)
        accuracy_plot_path = os.path.join(output_dir, 'prediction_accuracy.png')
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Global Active Power (kW)')
        plt.ylabel('Predicted Global Active Power (kW)')
        plt.title('Prediction vs Actual')
        plt.savefig(accuracy_plot_path)
        plt.close()
        logger.info(f"Saved prediction accuracy plot to {accuracy_plot_path}")
        
        # Feature Importance Plot
        feature_plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.figure(figsize=(10, 5))
        feature_importance.plot(x='feature', y='importance', kind='bar')
        plt.title('Feature Importance for Power Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(feature_plot_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {feature_plot_path}")
        
        # Hourly Consumption Plot
        hourly_consumption = energy_analyzer.hourly_consumption
        hourly_plot_path = os.path.join(output_dir, 'hourly_consumption.png')
        plt.figure(figsize=(12, 5))
        hourly_consumption.plot(kind='line', marker='o')
        plt.title('Average Energy Consumption by Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Average Global Active Power (kW)')
        plt.xticks(range(24))
        plt.grid(True)
        plt.savefig(hourly_plot_path)
        plt.close()
        logger.info(f"Saved hourly consumption plot to {hourly_plot_path}")

        logger.info("Analysis complete. Check the outputs directory for results.")

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure matplotlib doesn't try to open interactive windows
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    main()