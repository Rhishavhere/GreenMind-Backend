"""
Power Consumption Analysis Project Initialization
"""
import logging
import os
import sys

# Add the project root and src directories to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Configure project-wide logging
def configure_logging(log_level=logging.INFO, log_file='power_consumption.log'):
    """
    Configure logging for the entire project
    
    Args:
        log_level (int): Logging level (default: logging.INFO)
        log_file (str): Path to the log file
    """
    # Ensure log directory exists
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Full path for log file
    full_log_path = os.path.join(log_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(full_log_path),
            logging.StreamHandler()
        ]
    )

# Project configuration
PROJECT_CONFIG = {
    'name': 'Household Power Consumption Analysis',
    'version': '1.0.0',
    'data_dir': os.path.join(project_root, 'data'),
    'output_dir': os.path.join(project_root, 'outputs')
}

# Ensure output directories exist
os.makedirs(PROJECT_CONFIG['data_dir'], exist_ok=True)
os.makedirs(PROJECT_CONFIG['output_dir'], exist_ok=True)

# Optional: Import key modules for easy access
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_training import PowerConsumptionModel
from src.energy_analysis import EnergyAnalyzer
from src.visualization import EnergyVisualizer

# Configure logging when the module is imported
configure_logging()