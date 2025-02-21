import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")

# Read data
precip_df = pd.read_csv('aggragated_precip.csv')
inflow_df = pd.read_csv('water_years_base.csv')

# Convert dates
precip_df['Date'] = pd.to_datetime(precip_df['Month Year'], format='%b %Y')
inflow_df['Date'] = pd.to_datetime(inflow_df['Date'])

# Create monthly inflow averages
monthly_inflow = inflow_df.groupby(pd.Grouper(key='Date', freq='ME'))['Inflow'].mean().reset_index()
monthly_inflow['Date'] = monthly_inflow['Date'].dt.to_period('M').astype(str)

# Process each station separately to handle missing data
merged_df = monthly_inflow.copy()

for station in precip_df['Stn Name'].unique():
    # Get station data
    station_data = precip_df[precip_df['Stn Name'] == station].copy()
    station_data['Date'] = station_data['Date'].dt.to_period('M').astype(str)
    
    # Create station precipitation column
    station_data = station_data[['Date', 'Total Precip (in)']]
    station_data.columns = ['Date', f'{station}_Precip']
    
    # Merge with main dataframe
    merged_df = merged_df.merge(station_data, on='Date', how='left')

# Calculate correlations for different lags
max_lag = 6  # months
correlations = []

# Calculate correlations for each station
for station in precip_df['Stn Name'].unique():
    station_region = precip_df[precip_df['Stn Name'] == station]['CIMIS Region'].iloc[0]
    station_col = f'{station}_Precip'
    
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
            'Region': precip_df[precip_df['Stn Name'] == station]['CIMIS Region'].iloc[0],
            'Lag (months)': lag,
            'Correlation': corr
        })

corr_df = pd.DataFrame(correlations)

# Add month information for seasonal analysis
merged_df['Month'] = pd.to_datetime(merged_df['Date']).dt.month

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
pivot_corr['Region'] = [precip_df[precip_df['Stn Name'] == station]['CIMIS Region'].iloc[0] 
                       for station in pivot_corr.index]
pivot_corr = pivot_corr.sort_values(['Region', 0], ascending=[True, False])
pivot_corr = pivot_corr.drop('Region', axis=1)

sns.heatmap(pivot_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f', ax=ax1)
ax1.set_title('Station-Lag Correlation Heatmap\n(Sorted by Region and Correlation)')
ax1.set_xlabel('Lag (months)')

# 2. Regional correlation summary
ax2 = plt.subplot(2, 2, 2)
regional_corr = corr_df.groupby(['Region', 'Lag (months)'])['Correlation'].mean().unstack()
sns.heatmap(regional_corr, annot=True, cmap='RdYlBu', center=0, fmt='.2f', ax=ax2)
ax2.set_title('Regional Average Correlation by Lag')
ax2.set_xlabel('Lag (months)')

# 3. Monthly correlation patterns
ax3 = plt.subplot(2, 2, (3, 4))
monthly_patterns = []
for station in precip_df['Stn Name'].unique():
    station_col = f'{station}_Precip'
    if station_col in merged_df.columns:
        monthly_corr = merged_df.groupby('Month').apply(
            lambda x: x[station_col].corr(x['Inflow'])
        )
        monthly_patterns.append({
            'Station': station,
            'Region': precip_df[precip_df['Stn Name'] == station]['CIMIS Region'].iloc[0],
            'Monthly_Corr': monthly_corr
        })

for pattern in monthly_patterns:
    plt.plot(range(1, 13), pattern['Monthly_Corr'], 
             label=f"{pattern['Station']} ({pattern['Region']})",
             alpha=0.7, marker='o')

plt.grid(True, alpha=0.3)
plt.title('Monthly Correlation Patterns by Station')
plt.xlabel('Month')
plt.ylabel('Correlation with Inflow')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(range(1, 13))

plt.tight_layout()
plt.savefig('regional_precip_analysis.png', bbox_inches='tight', dpi=300)

# Remove the duplicate month addition and time series plot
print("\nBest Correlations by Station:")
for station in corr_df['Station'].unique():
    station_data = corr_df[corr_df['Station'] == station]
    best_lag = station_data.loc[station_data['Correlation'].idxmax()]
    print(f"\n{station} ({best_lag['Region']}):")
    print(f"Best correlation: {best_lag['Correlation']:.3f} at {best_lag['Lag (months)']} month lag")

print("\n\nBest Correlations by Region:")
for region in corr_df['Region'].unique():
    region_data = corr_df[corr_df['Region'] == region]
    best_lag = region_data.loc[region_data['Correlation'].idxmax()]
    print(f"\n{region}:")
    print(f"Best correlation: {best_lag['Correlation']:.3f} at {best_lag['Lag (months)']} month lag")
    print(f"Best station: {best_lag['Station']}")

# Additional analysis of seasonal patterns
print("\nSeasonal Analysis:")

for station in precip_df['Stn Name'].unique():
    station_col = f'{station}_Precip'
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
