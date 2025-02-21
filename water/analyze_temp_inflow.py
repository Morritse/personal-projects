import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")

# Read data
temp_df = pd.read_csv('aggragated_precip.csv')
inflow_df = pd.read_csv('water_years_base.csv')

# Convert dates
temp_df['Date'] = pd.to_datetime(temp_df['Month Year'], format='%b %Y')
inflow_df['Date'] = pd.to_datetime(inflow_df['Date'])

# Create monthly inflow averages
monthly_inflow = inflow_df.groupby(pd.Grouper(key='Date', freq='ME'))['Inflow'].mean().reset_index()
monthly_inflow['Date'] = monthly_inflow['Date'].dt.to_period('M').astype(str)

# Process each station separately to handle missing data
merged_df = monthly_inflow.copy()

for station in temp_df['Stn Name'].unique():
    # Get station data
    station_data = temp_df[temp_df['Stn Name'] == station].copy()
    station_data['Date'] = station_data['Date'].dt.to_period('M').astype(str)
    
    # Create station temperature column
    station_data = station_data[['Date', 'Avg Min Air Temp (F)']]
    station_data.columns = ['Date', f'{station}_Temp']
    
    # Merge with main dataframe
    merged_df = merged_df.merge(station_data, on='Date', how='left')

# Add month information for seasonal analysis
merged_df['Month'] = pd.to_datetime(merged_df['Date']).dt.month

# Calculate correlations for different lags
max_lag = 6  # months
correlations = []

# Calculate correlations for each station
for station in temp_df['Stn Name'].unique():
    station_region = temp_df[temp_df['Stn Name'] == station]['CIMIS Region'].iloc[0]
    station_col = f'{station}_Temp'
    
    # Skip if no data for this station
    if station_col not in merged_df.columns:
        continue
        
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = merged_df[station_col].corr(merged_df['Inflow'], min_periods=12)
        else:
            corr = merged_df[station_col].shift(lag).corr(merged_df['Inflow'], min_periods=12)
        
        correlations.append({
            'Station': station,
            'Region': temp_df[temp_df['Stn Name'] == station]['CIMIS Region'].iloc[0],
            'Lag (months)': lag,
            'Correlation': corr
        })

corr_df = pd.DataFrame(correlations)

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Correlation heatmap by station and lag
ax1 = plt.subplot(2, 2, 1)
pivot_corr = corr_df.pivot(
    index='Station',
    columns='Lag (months)',
    values='Correlation'
).dropna(how='all')

# Sort stations by region and maximum correlation
station_max_corr = pivot_corr.max(axis=1)
pivot_corr['Region'] = [temp_df[temp_df['Stn Name'] == station]['CIMIS Region'].iloc[0] 
                       for station in pivot_corr.index]
pivot_corr = pivot_corr.sort_values(['Region', 0], ascending=[True, False])
pivot_corr = pivot_corr.drop('Region', axis=1)

sns.heatmap(pivot_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f', ax=ax1)
ax1.set_title('Temperature-Inflow Correlation Heatmap\n(Sorted by Region and Correlation)')
ax1.set_xlabel('Lag (months)')

# 2. Regional correlation summary
ax2 = plt.subplot(2, 2, 2)
regional_corr = corr_df.groupby(['Region', 'Lag (months)'])['Correlation'].mean().unstack()
sns.heatmap(regional_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f', ax=ax2)
ax2.set_title('Regional Average Temperature Correlation by Lag')
ax2.set_xlabel('Lag (months)')

# 3. Temperature vs Inflow scatter plot
ax3 = plt.subplot(2, 2, 3)
for station in temp_df['Stn Name'].unique():
    station_col = f'{station}_Temp'
    if station_col in merged_df.columns:
        plt.scatter(merged_df[station_col], merged_df['Inflow'], 
                   alpha=0.3, label=station)
plt.title('Temperature vs Inflow Relationship')
plt.xlabel('Average Minimum Temperature (F)')
plt.ylabel('Inflow')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 4. Monthly correlation patterns
ax4 = plt.subplot(2, 2, 4)
monthly_patterns = []
for station in temp_df['Stn Name'].unique():
    station_col = f'{station}_Temp'
    if station_col in merged_df.columns:
        monthly_corr = merged_df.groupby('Month').apply(
            lambda x: x[station_col].corr(x['Inflow'])
        )
        monthly_patterns.append({
            'Station': station,
            'Region': temp_df[temp_df['Stn Name'] == station]['CIMIS Region'].iloc[0],
            'Monthly_Corr': monthly_corr
        })

for pattern in monthly_patterns:
    plt.plot(range(1, 13), pattern['Monthly_Corr'], 
             label=f"{pattern['Station']} ({pattern['Region']})",
             alpha=0.7, marker='o')

plt.grid(True, alpha=0.3)
plt.title('Monthly Temperature-Inflow Correlation Patterns')
plt.xlabel('Month')
plt.ylabel('Correlation with Inflow\n(+ = warmer → more inflow, - = warmer → less inflow)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(range(1, 13))

# Add horizontal line at zero
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('temp_inflow_analysis.png', bbox_inches='tight', dpi=300)

# Print detailed statistics
print("\nTemperature-Inflow Relationships by Station:")
for station in corr_df['Station'].unique():
    station_data = corr_df[corr_df['Station'] == station]
    best_lag = station_data.loc[station_data['Correlation'].idxmax()]
    print(f"\n{station} ({best_lag['Region']}):")
    relationship = "Higher temperatures → More inflow" if best_lag['Correlation'] > 0 else "Higher temperatures → Less inflow"
    print(f"Correlation: {best_lag['Correlation']:.3f} at {best_lag['Lag (months)']} month lag")
    print(f"Relationship: {relationship}")

print("\n\nTemperature-Inflow Relationships by Region:")
for region in corr_df['Region'].unique():
    region_data = corr_df[corr_df['Region'] == region]
    best_lag = region_data.loc[region_data['Correlation'].idxmax()]
    print(f"\n{region}:")
    relationship = "Higher temperatures → More inflow" if best_lag['Correlation'] > 0 else "Higher temperatures → Less inflow"
    print(f"Correlation: {best_lag['Correlation']:.3f} at {best_lag['Lag (months)']} month lag")
    print(f"Best station: {best_lag['Station']}")
    print(f"Relationship: {relationship}")

# Additional analysis of seasonal patterns
print("\nSeasonal Analysis:")
for station in temp_df['Stn Name'].unique():
    station_col = f'{station}_Temp'
    if station_col not in merged_df.columns:
        continue
        
    seasonal_corr = merged_df.groupby('Month').apply(
        lambda x: x[station_col].corr(x['Inflow'])
    )
    
    print(f"\n{station}:")
    print("Strongest correlations in months:")
    top_months = seasonal_corr.nlargest(3)
    for month, corr in top_months.items():
        print(f"Month {month}: {corr:.3f}")
