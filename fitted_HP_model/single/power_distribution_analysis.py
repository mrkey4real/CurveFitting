#!/usr/bin/env python3
"""
Outdoor Unit Power Distribution Analysis
=========================================

Analyze the distribution of outdoor unit power data to help select 
an appropriate threshold for filtering HVAC operation data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "merged_true_1min_data.csv"
POWER_COL = "Outdoor Unit [kW]"

def load_and_analyze_power_data():
    """Load and perform basic analysis of power data"""
    print("=== Loading and Analyzing Power Data ===")
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    power_data = df[POWER_COL]
    
    print(f"Total data points: {len(power_data):,}")
    print(f"Power range: {power_data.min():.6f} to {power_data.max():.3f} kW")
    print(f"Mean power: {power_data.mean():.3f} kW")
    print(f"Median power: {power_data.median():.3f} kW")
    print(f"Standard deviation: {power_data.std():.3f} kW")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  25th percentile: {power_data.quantile(0.25):.3f} kW")
    print(f"  50th percentile: {power_data.quantile(0.50):.3f} kW")
    print(f"  75th percentile: {power_data.quantile(0.75):.3f} kW")
    print(f"  90th percentile: {power_data.quantile(0.90):.3f} kW")
    print(f"  95th percentile: {power_data.quantile(0.95):.3f} kW")
    print(f"  99th percentile: {power_data.quantile(0.99):.3f} kW")
    
    return power_data

def analyze_threshold_impact(power_data, thresholds):
    """Analyze the impact of different thresholds"""
    print(f"\n=== Threshold Impact Analysis ===")
    
    total_points = len(power_data)
    
    print(f"{'Threshold (kW)':<15} {'Remaining Points':<15} {'Percentage':<12} {'Points Removed':<15}")
    print("-" * 65)
    
    for threshold in thresholds:
        remaining = (power_data > threshold).sum()
        percentage = 100 * remaining / total_points
        removed = total_points - remaining
        print(f"{threshold:<15.3f} {remaining:<15,} {percentage:<12.1f}% {removed:<15,}")
    
    return thresholds

def create_distribution_plots(power_data):
    """Create comprehensive distribution plots"""
    print(f"\n=== Creating Distribution Plots ===")
    
    # Set up the plot style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall histogram
    plt.subplot(2, 4, 1)
    plt.hist(power_data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Power (kW)')
    plt.ylabel('Frequency')
    plt.title('Overall Power Distribution')
    plt.grid(True, alpha=0.3)
    
    # 2. Log-scale histogram (for better visibility of low values)
    plt.subplot(2, 4, 2)
    plt.hist(power_data, bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Power (kW)')
    plt.ylabel('Frequency (Log Scale)')
    plt.title('Power Distribution (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. Zoomed-in histogram for low power values (0-1 kW)
    plt.subplot(2, 4, 3)
    low_power = power_data[power_data <= 1.0]
    plt.hist(low_power, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Power (kW)')
    plt.ylabel('Frequency')
    plt.title('Low Power Distribution (0-1 kW)')
    plt.grid(True, alpha=0.3)
    
    # 4. Box plot
    plt.subplot(2, 4, 4)
    plt.boxplot(power_data, vert=True)
    plt.ylabel('Power (kW)')
    plt.title('Power Distribution Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution
    plt.subplot(2, 4, 5)
    sorted_power = np.sort(power_data)
    cumulative = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
    plt.plot(sorted_power, cumulative, linewidth=2, color='red')
    plt.xlabel('Power (kW)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # 6. Threshold analysis visualization
    plt.subplot(2, 4, 6)
    thresholds = np.arange(0, 2.1, 0.1)
    remaining_percentages = []
    
    for threshold in thresholds:
        remaining = (power_data > threshold).sum()
        percentage = 100 * remaining / len(power_data)
        remaining_percentages.append(percentage)
    
    plt.plot(thresholds, remaining_percentages, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Threshold (kW)')
    plt.ylabel('Remaining Data (%)')
    plt.title('Data Retention vs Threshold')
    plt.grid(True, alpha=0.3)
    
    # Add common threshold lines
    common_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for thresh in common_thresholds:
        if thresh <= thresholds.max():
            idx = np.argmin(np.abs(thresholds - thresh))
            plt.axvline(x=thresh, color='red', linestyle='--', alpha=0.7)
            plt.text(thresh, remaining_percentages[idx] + 5, f'{thresh:.1f}', 
                    rotation=90, ha='center', va='bottom')
    
    # 7. Power vs Time (sample)
    plt.subplot(2, 4, 7)
    # Sample every 60 points for visualization (1 hour intervals)
    sample_indices = range(0, len(power_data), 60)
    sample_power = power_data.iloc[sample_indices]
    plt.plot(sample_power.index, sample_power.values, alpha=0.7, linewidth=0.5)
    plt.xlabel('Time Index (Hours)')
    plt.ylabel('Power (kW)')
    plt.title('Power Time Series (Sampled)')
    plt.grid(True, alpha=0.3)
    
    # 8. Power density plot
    plt.subplot(2, 4, 8)
    # Create density plot
    density = stats.gaussian_kde(power_data)
    x_range = np.linspace(power_data.min(), power_data.max(), 1000)
    plt.plot(x_range, density(x_range), linewidth=2, color='purple')
    plt.fill_between(x_range, density(x_range), alpha=0.3, color='purple')
    plt.xlabel('Power (kW)')
    plt.ylabel('Density')
    plt.title('Power Density Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("Distribution plots saved as: power_distribution_analysis.png")
    plt.show()

def suggest_thresholds(power_data):
    """Suggest appropriate thresholds based on data analysis"""
    print(f"\n=== Threshold Recommendations ===")
    
    # Calculate various statistics-based thresholds
    mean_power = power_data.mean()
    std_power = power_data.std()
    median_power = power_data.median()
    
    # Different threshold strategies
    print(f"Statistical-based suggestions:")
    print(f"  Mean: {mean_power:.3f} kW")
    print(f"  Median: {median_power:.3f} kW")
    print(f"  Mean - 1*STD: {max(0, mean_power - std_power):.3f} kW")
    print(f"  Mean - 2*STD: {max(0, mean_power - 2*std_power):.3f} kW")
    
    # Percentile-based thresholds
    print(f"\nPercentile-based suggestions:")
    print(f"  5th percentile: {power_data.quantile(0.05):.3f} kW")
    print(f"  10th percentile: {power_data.quantile(0.10):.3f} kW")
    print(f"  15th percentile: {power_data.quantile(0.15):.3f} kW")
    print(f"  20th percentile: {power_data.quantile(0.20):.3f} kW")
    
    # Gap-based analysis (find natural breaks in the data)
    print(f"\nGap analysis (looking for natural breaks):")
    sorted_power = np.sort(power_data.unique())
    gaps = np.diff(sorted_power)
    large_gaps_idx = np.where(gaps > np.percentile(gaps, 95))[0]
    
    print(f"  Large gaps found at:")
    for idx in large_gaps_idx[:5]:  # Show top 5 gaps
        if sorted_power[idx] < 2.0:  # Only show reasonable thresholds
            print(f"    {sorted_power[idx]:.3f} kW -> {sorted_power[idx+1]:.3f} kW (gap: {gaps[idx]:.3f})")
    
    # Practical recommendations
    print(f"\nPractical recommendations:")
    print(f"  Conservative (keep most data): 0.1-0.2 kW")
    print(f"  Moderate (balance quality/quantity): 0.3-0.5 kW")
    print(f"  Aggressive (high-quality only): 0.8-1.0 kW")

def create_detailed_threshold_table(power_data):
    """Create a detailed table showing threshold impacts"""
    print(f"\n=== Detailed Threshold Analysis Table ===")
    
    thresholds = np.arange(0.0, 2.1, 0.05)
    results = []
    
    for threshold in thresholds:
        filtered_data = power_data[power_data > threshold]
        remaining_count = len(filtered_data)
        remaining_pct = 100 * remaining_count / len(power_data)
        
        if remaining_count > 0:
            filtered_mean = filtered_data.mean()
            filtered_min = filtered_data.min()
            filtered_max = filtered_data.max()
        else:
            filtered_mean = filtered_min = filtered_max = 0
        
        results.append({
            'Threshold': threshold,
            'Remaining_Count': remaining_count,
            'Remaining_Percent': remaining_pct,
            'Filtered_Mean': filtered_mean,
            'Filtered_Min': filtered_min,
            'Filtered_Max': filtered_max
        })
    
    # Convert to DataFrame for nice display
    df_results = pd.DataFrame(results)
    
    # Show key thresholds
    key_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0]
    print(f"{'Threshold':<10} {'Count':<8} {'Percent':<8} {'Mean':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)
    
    for thresh in key_thresholds:
        if thresh <= thresholds.max():
            row = df_results[df_results['Threshold'] == thresh].iloc[0]
            print(f"{row['Threshold']:<10.1f} {row['Remaining_Count']:<8,} "
                  f"{row['Remaining_Percent']:<8.1f}% {row['Filtered_Mean']:<8.3f} "
                  f"{row['Filtered_Min']:<8.3f} {row['Filtered_Max']:<8.3f}")

def main():
    print("=== Outdoor Unit Power Distribution Analysis ===")
    print("Analyzing power data to determine optimal threshold for HVAC operation detection")
    print()
    
    try:
        # Load and analyze data
        power_data = load_and_analyze_power_data()
        
        # Analyze threshold impacts
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
        analyze_threshold_impact(power_data, test_thresholds)
        
        # Create distribution plots
        create_distribution_plots(power_data)
        
        # Suggest thresholds
        suggest_thresholds(power_data)
        
        # Detailed threshold table
        create_detailed_threshold_table(power_data)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Check the generated plot 'power_distribution_analysis.png' to visualize the distribution.")
        print(f"Based on the analysis above, select an appropriate threshold for your needs.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 