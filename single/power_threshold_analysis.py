#!/usr/bin/env python3
"""
Power Threshold Analysis for HVAC Operation Detection
====================================================

Simple and effective analysis to help select optimal threshold
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "../data/single/final_merged_weather_egauge.csv"
POWER_COL = "Outdoor Unit [kW]"

def main():
    print("=== Power Threshold Analysis ===")
    
    # Load data
    df = pd.read_csv(DATA_FILE)
    power_data = df[POWER_COL]
    
    print(f"Total data points: {len(power_data):,}")
    print(f"Power range: {power_data.min():.6f} to {power_data.max():.3f} kW")
    print(f"Mean power: {power_data.mean():.3f} kW")
    print(f"Median power: {power_data.median():.6f} kW")
    
    # Key statistics
    print(f"\nKey Percentiles:")
    percentiles = [0, 5, 10, 15, 20, 25, 30, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        value = power_data.quantile(p/100)
        print(f"  {p:2d}th percentile: {value:.3f} kW")
    
    # Threshold analysis
    print(f"\nThreshold Impact Analysis:")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"{'Threshold':<10} {'Remaining':<12} {'Percentage':<12} {'Removed':<12}")
    print("-" * 50)
    
    for thresh in thresholds:
        remaining = (power_data > thresh).sum()
        percentage = 100 * remaining / len(power_data)
        removed = len(power_data) - remaining
        print(f"{thresh:<10.2f} {remaining:<12,} {percentage:<12.1f}% {removed:<12,}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall histogram
    axes[0, 0].hist(power_data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Power (kW)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Power Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log scale histogram
    axes[0, 1].hist(power_data[power_data > 0], bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Power (kW)')
    axes[0, 1].set_ylabel('Frequency (Log Scale)')
    axes[0, 1].set_title('Power Distribution (Log Scale, >0)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Low power zoom (0-1 kW)
    low_power = power_data[power_data <= 1.0]
    axes[0, 2].hist(low_power, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Power (kW)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Low Power Distribution (0-1 kW)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_power = np.sort(power_data)
    cumulative = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
    axes[1, 0].plot(sorted_power, cumulative, linewidth=2, color='red')
    axes[1, 0].set_xlabel('Power (kW)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add threshold lines
    common_thresholds = [0.1, 0.2, 0.4, 0.5, 0.8]
    for thresh in common_thresholds:
        cumul_at_thresh = np.sum(sorted_power <= thresh) / len(sorted_power)
        axes[1, 0].axvline(x=thresh, color='blue', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=cumul_at_thresh, color='blue', linestyle='--', alpha=0.7)
        axes[1, 0].text(thresh, cumul_at_thresh + 0.05, f'{thresh:.1f}kW\n{cumul_at_thresh:.1%}', 
                       ha='center', va='bottom', fontsize=8)
    
    # 5. Data retention vs threshold
    test_thresholds = np.arange(0, 2.1, 0.05)
    remaining_percentages = []
    
    for threshold in test_thresholds:
        remaining = (power_data > threshold).sum()
        percentage = 100 * remaining / len(power_data)
        remaining_percentages.append(percentage)
    
    axes[1, 1].plot(test_thresholds, remaining_percentages, linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_xlabel('Threshold (kW)')
    axes[1, 1].set_ylabel('Remaining Data (%)')
    axes[1, 1].set_title('Data Retention vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add threshold recommendations
    recommended_thresholds = [0.1, 0.2, 0.4, 0.5, 0.8]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    
    for thresh, color in zip(recommended_thresholds, colors):
        if thresh <= max(test_thresholds):
            idx = np.argmin(np.abs(test_thresholds - thresh))
            axes[1, 1].axvline(x=thresh, color=color, linestyle='--', alpha=0.7)
            axes[1, 1].text(thresh, remaining_percentages[idx] + 2, f'{thresh:.1f}kW', 
                           rotation=90, ha='center', va='bottom', color=color, fontweight='bold')
    
    # 6. Power vs time (sampled)
    sample_indices = range(0, len(power_data), 60)  # Every hour
    sample_power = power_data.iloc[sample_indices]
    sample_time = np.arange(len(sample_power))
    
    axes[1, 2].plot(sample_time, sample_power.values, alpha=0.7, linewidth=0.5)
    axes[1, 2].set_xlabel('Time (Hours)')
    axes[1, 2].set_ylabel('Power (kW)')
    axes[1, 2].set_title('Power Time Series (Hourly Samples)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add threshold lines
    for thresh in recommended_thresholds[:3]:  # Show only first 3 to avoid clutter
        axes[1, 2].axhline(y=thresh, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].text(0, thresh, f'{thresh:.1f}kW', va='bottom', ha='left', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('power_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved as: power_threshold_analysis.png")
    plt.show()

if __name__ == "__main__":
    main() 