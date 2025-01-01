# Al-powered solution to enhance energy efficiency and sustainability in residential and commercial buildings

**Introduction**

This project aims to develop a machine learning model to accurately predict household power consumption and identify peak usage times. By leveraging historical data and advanced algorithms, we can enable energy consumers to make informed decisions, optimize energy usage, and contribute to a more sustainable future.

**Dataset**

Link : [https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption]

The dataset used in this project comprises historical household power consumption data, encompassing the following features:

* **Timestamp:** Time of the measurement
* **Global_active_power:** Total active power consumed
* **Global_reactive_power:** Total reactive power consumed
* **Voltage:** Voltage in volts
* **Global_intensity:** Total current intensity
* **Sub_metering_1:** Energy consumption of meter 1 (kitchen)
* **Sub_metering_2:** Energy consumption of meter 2 (laundry room)
* **Sub_metering_3:** Energy consumption of meter 3 (bedroom)

# Backend Project Structure and File Overview

This document provides an overview of the backend files and their roles in the project. The project is structured to handle data loading, preprocessing, model training, energy analysis, and visualization in a modular and organized manner.

## 1. Project Setup and Configuration

*   **`Backend/src/__init__.py`**:
    *   **Purpose**: This file initializes the project environment.
    *   **Key Functions**:
        *   Sets up the Python path to ensure all modules can be imported correctly.
        *   Configures logging for the entire project, including setting the log level, format, and output file.
        *   Defines project-wide configurations like the project name, version, data directory, and output directory.
        *   Creates the data and output directories if they don't exist.
        *   Imports key modules, making them easily accessible throughout the project.
    *   **Role**: Acts as the central configuration and setup file for the backend.

## 2. Data Handling

*   **`Backend/src/data_loader.py`**:
    *   **Purpose**: Responsible for loading the dataset from a CSV file.
    *   **Key Functions**:
        *   `__init__`: Initializes the data loader with the file path and sets up a logger.
        *   `load_data`: Reads the CSV file, parses date and time columns, converts numeric columns, handles missing values, and returns a Pandas DataFrame.
    *   **Role**: The first step in the data pipeline, ensuring data is read and cleaned before further processing.

*   **`Backend/src/preprocessing.py`**:
    *   **Purpose**: Prepares the data for model training.
    *   **Key Functions**:
        *   `__init__`: Initializes the preprocessor with the input DataFrame.
        *   `extract_time_features`: Extracts time-based features (hour, day of the week, month) from the datetime column.
        *   `create_features`: Combines time features, calculates total sub-metering, and selects the features and target variable for the model.
        *   `prepare_train_test_split`: Splits the data into training and testing sets, scales the features using `StandardScaler`, and returns the scaled data along with the scaler.
    *   **Role**: Transforms raw data into a format suitable for machine learning models.

## 3. Model Training and Evaluation

*   **`Backend/src/model_training.py`**:
    *   **Purpose**: Trains and evaluates the machine learning model.
    *   **Key Functions**:
        *   `__init__`: Initializes the `RandomForestRegressor` model with specified parameters.
        *   `train`: Trains the model using the training data.
        *   `evaluate`: Evaluates the model's performance using metrics like MSE, MAE, and R2.
        *   `get_feature_importance`: Extracts and returns the feature importances from the trained model.
        *   `predict`: Makes predictions on new data using the trained model.
    *   **Role**: The core of the machine learning process, responsible for training and assessing the model.

## 4. Energy Analysis

*   **`Backend/src/energy_analysis.py`**:
    *   **Purpose**: Analyzes energy consumption patterns and provides recommendations.
    *   **Key Functions**:
        *   `__init__`: Initializes the analyzer with the input DataFrame.
        *   `identify_peak_usage_times`: Identifies and prints the hours with the highest average energy consumption.
        *   `analyze_sub_metering_impact`: Calculates and prints the percentage of energy consumption by each sub-metering category.
        *   `generate_energy_recommendations`: Generates energy-saving recommendations based on sub-metering analysis and peak usage times.
        *   `generate_sustainability_report`: Combines all analysis steps and prints a comprehensive sustainability report.
    *   **Role**: Provides insights into energy usage patterns and suggests ways to improve efficiency.

## 5. Visualization

*   **`Backend/src/visualization.py`**:
    *   **Purpose**: Creates visualizations to help understand the data and model results.
    *   **Key Functions**:
        *   `plot_prediction_accuracy`: Generates a scatter plot of predicted vs. actual values.
        *   `plot_feature_importance`: Creates a bar plot of feature importances.
        *   `plot_hourly_consumption`: Generates a line plot of average energy consumption by hour.
    *   **Role**: Provides visual representations of the data and model performance.

## 6. Main Application Logic

*   **`Backend/main.py`**:
    *   **Purpose**: Orchestrates the entire workflow of the project.
    *   **Key Functions**:
        *   `main`:
            *   Loads data using `DataLoader`.
            *   Preprocesses data using `DataPreprocessor`.
            *   Trains the model using `PowerConsumptionModel`.
            *   Evaluates the model and logs the metrics.
            *   Analyzes feature importance.
            *   Performs energy analysis using `EnergyAnalyzer`.
            *   Generates and saves visualizations using `matplotlib`.
    *   **Role**: The entry point of the application, coordinating all the other modules to perform the analysis.

## 7. API Endpoint

*   **`Backend/api.py`**:
    *   **Purpose**: Provides an API endpoint for making predictions using the trained model.
    *   **Key Functions**:
        *   Loads the trained model from a pickle file.
        *   Defines a `/predict` endpoint that accepts POST requests with input data.
        *   Uses the model to make predictions and returns the results as JSON.
    *   **Role**: Exposes the model's prediction capabilities through a web API.

## Flow of Operations

1.  **Initialization**: `__init__.py` sets up the environment, logging, and configurations.
2.  **Data Loading**: `DataLoader` loads and cleans the data.
3.  **Preprocessing**: `DataPreprocessor` extracts features, scales the data, and splits it into training and testing sets.
4.  **Model Training**: `PowerConsumptionModel` trains the model on the training data.
5.  **Model Evaluation**: `PowerConsumptionModel` evaluates the model's performance on the test data.
6.  **Feature Importance**: `PowerConsumptionModel` extracts feature importances.
7.  **Energy Analysis**: `EnergyAnalyzer` analyzes energy consumption patterns and generates recommendations.
8.  **Visualization**: `EnergyVisualizer` (or `main.py` directly) creates plots for prediction accuracy, feature importance, and hourly consumption.
9.  **API**: `api.py` exposes the trained model for predictions via a web API.
10. **Main Execution**: `main.py` orchestrates all the above steps.

This structure provides a clear separation of concerns, making the project modular and easier to maintain. Each file has a specific role in the overall process, from data loading to model deployment.