import numpy as np
import pandas as pd

class EnergyAnalyzer:

    def __init__(self, df):

        self.df = df
        self.hourly_consumption = None
        self.sub_metering_breakdown = None
    
    def identify_peak_usage_times(self):

        self.hourly_consumption = self.df.groupby(
            [self.df['Datetime'].dt.hour, self.df['Datetime'].dt.minute]
        )['Global_active_power'].mean()
        
        peak_times = self.hourly_consumption.nlargest(3)
        
        print("Peak Energy Consumption Times:")
        for (hour, minute), consumption in peak_times.items():
            print(f"Time {hour:02d}:{minute:02d} - Average Consumption: {consumption:.2f} kW")
        
        return peak_times
    
    def analyze_sub_metering_impact(self):

        sub_metering_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        total_energy = self.df[sub_metering_columns].sum()
        self.sub_metering_breakdown = total_energy / total_energy.sum() * 100
        
        print("\nEnergy Consumption by Appliance Category:")
        for category, percentage in self.sub_metering_breakdown.items():
            print(f"{category}: {percentage:.2f}%")
        
        return self.sub_metering_breakdown
    
    def generate_energy_recommendations(self):

        recommendations = []
        
        # Kitchen Appliances
        if self.sub_metering_breakdown['Sub_metering_1'] > 25:
            recommendations.append(
                "Kitchen Energy-Saving Tips:\n"
                "- Use dishwasher only when fully loaded\n"
                "- Unplug small appliances when not in use\n"
                "- Use energy-efficient cooking methods"
            )
        
        # Laundry and Refrigeration
        if self.sub_metering_breakdown['Sub_metering_2'] > 25:
            recommendations.append(
                "Laundry and Refrigeration Efficiency:\n"
                "- Wash clothes in cold water\n"
                "- Clean refrigerator coils regularly\n"
                "- Ensure proper refrigerator door sealing"
            )
        
        # HVAC and Water Heating
        if self.sub_metering_breakdown['Sub_metering_3'] > 25:
            recommendations.append(
                "HVAC and Water Heating Optimization:\n"
                "- Lower water heater temperature\n"
                "- Use programmable thermostat\n"
                "- Improve home insulation\n"
                "- Regular HVAC system maintenance"
            )
        
        # Peak Hour Recommendations
        if self.hourly_consumption is not None:
            peak_hours = self.hourly_consumption.nlargest(3)
            peak_hour_tips = [
                f"Reduce energy usage during peak hours ({', '.join(map(str, peak_hours.index))})",
                "- Shift energy-intensive tasks to off-peak hours",
                "- Use timer-based appliances during low-consumption periods"
            ]
            recommendations.append("\n".join(peak_hour_tips))
        
        return recommendations
    
    def generate_sustainability_report(self):

        print("\n===== ENERGY SUSTAINABILITY REPORT =====")
        
        # Peak Usage Analysis
        self.identify_peak_usage_times()
        
        # Sub-Metering Breakdown
        self.analyze_sub_metering_impact()
        
        # Generate Recommendations
        recommendations = self.generate_energy_recommendations()
        
        print("\nRecommendations:")
        for recommendation in recommendations:
            print(f"\n{recommendation}")
        
        # Potential Savings Estimation
        estimated_savings = 0.2 * self.df['Global_active_power'].mean()
        print(f"\nPotential Energy Savings: {estimated_savings:.2f} kW")
        print("Estimated Annual Cost Reduction: ${:.2f}".format(
            estimated_savings * 365 * 0.12  # Assuming $0.12 per kWh
        ))
    
    def save_hourly_consumption(self, filepath):
        if self.hourly_consumption is not None:
            self.hourly_consumption.to_csv(filepath, header=True)
            print(f"Hourly consumption data saved to {filepath}")
        else:
            print("Hourly consumption data not yet generated.")