import pandas as pd
import logging

class DataLoader:

    def __init__(self, filepath):
 
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)
    
    def load_data(self):

        try:
            # Read CSV and parse datetime
            df = pd.read_csv(self.filepath, parse_dates=[['Date', 'Time']])
            
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
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise