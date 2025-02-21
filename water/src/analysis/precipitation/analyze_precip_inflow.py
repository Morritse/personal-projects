import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from scipy.interpolate import CubicSpline

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))

# Read precipitation data
precip_df = pd.read_csv('precipitation.csv')

# Melt precipitation data to get it in long format
precip_melted = pd.melt(precip_df, 
                        id_vars=['Year'], 
                        value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        var_name='Month', 
                        value_name='Precipitation')

# Convert month names to numbers
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
precip_melted['Month'] = precip_melted['Month'].map(month_map)

# Create datetime index for precipitation
precip_melted['Date'] = pd.to_datetime(precip_melted[['Year', 'Month']].assign(Day=1))

# Read inflow data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Filter both datasets to 2010-2019
precip_melted = precip_melted[(precip_melted['Date'].dt.year >= 2010) & 
                             (precip_melted['Date'].dt.year <= 2019)]
df = df[(df['Date'].dt.year >= 2010) & (df['Date'].dt.year <= 2019)]

# Process inflow data with rolling average
df['Year'] = df['Date'].dt.year
df['DayOfYear'] = df['Date'].dt.dayofyear

# Create smoother curves
def create_smooth_curves(precip_data, inflow_data, year=None):
    # Smooth precipitation using cubic spline
    # Add first point to end to create closed loop
    x_precip = np.append(precip_data['Month'].values, 13)
    y_precip = np.append(precip_data['Precipitation'].values, 
                        precip_data['Precipitation'].values[0])
    
    # Create smooth curve
    x_smooth_precip = np.linspace(1, 12, 365)
    cs_precip = CubicSpline(x_precip, y_precip)
    smooth_precip = cs_precip(x_smooth_precip)
    
    # Smooth inflow using rolling average
    if year is not None:
        year_inflow = inflow_data[inflow_data['Year'] == year]
    else:
        year_inflow = inflow_data
    
    # Sort by day of year and apply rolling average
    daily_inflow = year_inflow.sort_values('DayOfYear')
    smooth_inflow = daily_inflow['Inflow'].rolling(window=14, center=True).mean()
    
    return x_smooth_precip, smooth_precip, daily_inflow['DayOfYear'], smooth_inflow

# Create figure with subplots for each year
fig = plt.figure(figsize=(20, 25))
years = sorted(precip_melted['Year'].unique())

# First plot: Overall average patterns
ax_avg = plt.subplot(6, 2, 1)
avg_precip = precip_melted.groupby('Month')['Precipitation'].mean().reset_index()

# Get smooth curves for average pattern
x_smooth_precip, smooth_precip, x_inflow, smooth_inflow = create_smooth_curves(
    avg_precip, df)

# Scale inflow to match precipitation range for better visualization
inflow_scale = smooth_precip.max() / smooth_inflow.max()
scaled_inflow = smooth_inflow * inflow_scale

# Plot smoothed curves
ax_avg.plot(x_smooth_precip, smooth_precip, color='blue', label='Precipitation', alpha=0.7)
ax_avg_twin = ax_avg.twinx()
ax_avg_twin.plot(x_inflow * 12/365, smooth_inflow, color='green', label='Inflow', alpha=0.7)

# Add original points for precipitation
ax_avg.scatter(avg_precip['Month'], avg_precip['Precipitation'], 
               color='blue', alpha=0.5, s=30)

ax_avg.set_title('Average Monthly Patterns (2010-2019)')
ax_avg.set_xlabel('Month')
ax_avg.set_ylabel('Precipitation (inches)', color='blue')
ax_avg_twin.set_ylabel('Inflow', color='green')
ax_avg.tick_params(axis='y', labelcolor='blue')
ax_avg_twin.tick_params(axis='y', labelcolor='green')
ax_avg.set_xticks(range(1, 13))

# Add legends
lines1, labels1 = ax_avg.get_legend_handles_labels()
lines2, labels2 = ax_avg_twin.get_legend_handles_labels()
ax_avg.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Create individual year plots
for i, year in enumerate(years, 2):
    ax = plt.subplot(6, 2, i)
    
    # Get data for this year
    year_precip = precip_melted[precip_melted['Year'] == year].copy()
    
    # Get smooth curves for this year
    x_smooth_precip, smooth_precip, x_inflow, smooth_inflow = create_smooth_curves(
        year_precip, df, year)
    
    # Scale inflow to match precipitation range for this year
    year_inflow_scale = smooth_precip.max() / smooth_inflow.max()
    year_scaled_inflow = smooth_inflow * year_inflow_scale
    
    # Calculate month-to-month changes
    precip_changes = np.diff(smooth_precip)
    inflow_changes = np.diff(smooth_inflow)
    
    # Find regions where inflow increases but precipitation doesn't
    inflow_increase = np.where(inflow_changes > 0)[0]
    precip_decrease = np.where(precip_changes <= 0)[0]
    interesting_regions = np.intersect1d(inflow_increase, precip_decrease)
    
    # Plot smoothed curves
    ax.plot(x_smooth_precip, smooth_precip, color='blue', label='Precipitation', alpha=0.7)
    ax_twin = ax.twinx()
    ax_twin.plot(x_inflow * 12/365, smooth_inflow, color='green', label='Inflow', alpha=0.7)
    
    # Add original points for precipitation
    ax.scatter(year_precip['Month'], year_precip['Precipitation'], 
               color='blue', alpha=0.5, s=30)
    
    # Highlight interesting regions
    for idx in interesting_regions:
        month = x_smooth_precip[idx]
        if idx > 0 and idx < len(smooth_precip) - 1:  # Avoid edges
            ax.axvspan(month, month + 0.1, color='red', alpha=0.2)
    
    ax.set_title(f'Monthly Patterns - {year}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Precipitation (inches)', color='blue')
    ax_twin.set_ylabel('Inflow', color='green')
    ax.tick_params(axis='y', labelcolor='blue')
    ax_twin.tick_params(axis='y', labelcolor='green')
    ax.set_xticks(range(1, 13))
    
    # Add legends for first year plot only
    if i == 2:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
fig.suptitle('Precipitation and Inflow Patterns by Year', y=1.02, fontsize=16)

plt.tight_layout()
plt.savefig('precip_inflow_comparison.png', bbox_inches='tight')

# Calculate and print timing statistics
print("\nTiming Analysis:")

# Monthly precipitation stats
monthly_precip = precip_melted.groupby('Month')['Precipitation'].mean()
print("\nMonths with highest precipitation (average):")
print(monthly_precip.nlargest(3))

# Monthly inflow stats
monthly_inflow = df.groupby(df['Date'].dt.month)['Inflow'].mean()
print("\nMonths with highest inflow (average):")
print(monthly_inflow.nlargest(3))

# Calculate average lag to peak
precip_peak_month = monthly_precip.idxmax()
inflow_peak_month = monthly_inflow.idxmax()
lag_months = (inflow_peak_month - precip_peak_month) % 12
print(f"\nAverage lag from peak precipitation to peak inflow: {lag_months} months")

# Analyze counter-intuitive patterns
print("\nCounter-intuitive Pattern Analysis:")
monthly_changes = pd.DataFrame({
    'Precip_Change': monthly_precip.diff(),
    'Inflow_Change': monthly_inflow.diff()
})

# Count months where inflow increases but precipitation decreases
counter_pattern = monthly_changes[
    (monthly_changes['Inflow_Change'] > 0) & 
    (monthly_changes['Precip_Change'] <= 0)
]
print(f"\nMonths where inflow increases despite decreasing precipitation: {len(counter_pattern)}")
print("\nTypically occurs in:")
month_counts = counter_pattern.groupby(counter_pattern.index).size()
for month, count in month_counts.nlargest(3).items():
    print(f"Month {month}: {count} occurrences")
