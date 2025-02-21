import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Process FNF data
fnf_raw = pd.read_csv('fnf2010_2020.csv', header=None)
year_labels = list(range(2010, 2021))  # 2010-2020
fnf_raw.columns = year_labels

# Calculate yearly flow volumes (sum of daily flows)
yearly_volumes = {}
for year in year_labels:
    yearly_volumes[year] = fnf_raw[year].sum()  # Total volume of flow

# Process snow data
stations = ['GRM', 'HNT', 'KSP', 'VLC']
melt_analysis = pd.DataFrame(index=year_labels)

for station in stations:
    # Read data
    df = pd.read_csv(f'{station}.csv')
    snow_df = pd.DataFrame({
        'Date': pd.to_datetime(df['DATE TIME'].str[:8], format='%Y%m%d'),
        'Water_Level': pd.to_numeric(df['VALUE'], errors='coerce')
    })
    snow_df['Year'] = snow_df['Date'].dt.year
    snow_df['Month'] = snow_df['Date'].dt.month
    
    # Calculate metrics for each year
    yearly_stats = []
    for year in year_labels:
        year_data = snow_df[snow_df['Year'] == year].copy()
        if len(year_data) > 0:
            # Find peak snow and when it occurred
            peak_idx = year_data['Water_Level'].idxmax()
            peak_date = year_data.loc[peak_idx, 'Date']
            peak_level = year_data.loc[peak_idx, 'Water_Level']
            
            # Calculate melt after peak
            post_peak = year_data[year_data['Date'] >= peak_date].copy()
            post_peak['Melt_Rate'] = -post_peak['Water_Level'].diff()
            
            # Get total melt (peak to minimum)
            min_level = post_peak['Water_Level'].min()
            total_melt = peak_level - min_level
            
            # Calculate melt volume (area under the melt curve)
            melt_volume = post_peak['Melt_Rate'].clip(lower=0).sum()  # Only count positive melt
            
            yearly_stats.append({
                'Year': year,
                'Peak_Level': peak_level,
                'Total_Melt': total_melt,
                'Melt_Volume': melt_volume,
                'Peak_Date': peak_date
            })
    
    yearly_df = pd.DataFrame(yearly_stats)
    yearly_df.set_index('Year', inplace=True)
    
    # Add to main analysis DataFrame
    melt_analysis[f'{station}_Total_Melt'] = yearly_df['Total_Melt']
    melt_analysis[f'{station}_Melt_Volume'] = yearly_df['Melt_Volume']
    melt_analysis[f'{station}_Peak_Date'] = yearly_df['Peak_Date']

# Add flow volumes
melt_analysis['Flow_Volume'] = pd.Series(yearly_volumes)

# Create figure
plt.figure(figsize=(15, 15))

# Plot 1: Total Melt Volume vs Flow Volume
plt.subplot(3, 1, 1)
correlations = {}

for station in stations:
    x = pd.to_numeric(melt_analysis[f'{station}_Total_Melt'], errors='coerce')
    y = pd.to_numeric(melt_analysis['Flow_Volume'], errors='coerce')
    mask = ~(np.isnan(x) | np.isnan(y))
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(x[mask], y[mask])
    correlations[station] = (corr, p_value)
    
    plt.scatter(x, y, alpha=0.7, label=f'{station} (r={corr:.3f})')
    
    # Add trend line
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    plt.plot(x[mask], p(x[mask]), '--', alpha=0.5)

plt.title('Total Flow Volume vs Total Snow Melt by Station')
plt.xlabel('Total Seasonal Melt (inches)')
plt.ylabel('Total Flow Volume (cfs-days)')
plt.legend()

# Plot 2: Cumulative analysis
plt.subplot(3, 1, 2)

# Calculate cumulative flow and melt
example_year = 2017  # Year with high snow
year_data = fnf_raw[example_year].copy()
year_data.index = pd.date_range(start=f'{example_year-1}-10-01', periods=len(year_data))
cumulative_flow = year_data.cumsum()

plt.plot(year_data.index, cumulative_flow, 'k-', label='Cumulative Flow', linewidth=2)

for station in stations:
    station_data = snow_df[snow_df['Year'] == example_year].copy()
    if len(station_data) > 0:
        station_data.set_index('Date', inplace=True)
        station_data['Melt'] = -station_data['Water_Level'].diff()
        station_data['Cumulative_Melt'] = station_data['Melt'].clip(lower=0).cumsum()
        plt.plot(station_data.index, station_data['Cumulative_Melt'] * 1000,  # Scale for visibility
                '--', label=f'{station} Cumulative Melt', alpha=0.7)

plt.title(f'{example_year} Cumulative Flow and Snow Melt')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()

# Plot 3: Year-by-year volume comparison
plt.subplot(3, 1, 3)

# Normalize values for comparison
normalized_data = pd.DataFrame(index=year_labels)
normalized_data['Flow_Volume'] = pd.to_numeric(melt_analysis['Flow_Volume'], errors='coerce')
normalized_data['Flow_Volume'] = (normalized_data['Flow_Volume'] - normalized_data['Flow_Volume'].mean()) / normalized_data['Flow_Volume'].std()

for station in stations:
    col = f'{station}_Total_Melt'
    normalized_data[col] = pd.to_numeric(melt_analysis[col], errors='coerce')
    normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()

plt.plot(year_labels, normalized_data['Flow_Volume'], 'k-', label='Flow Volume', linewidth=2)
for station in stations:
    plt.plot(year_labels, normalized_data[f'{station}_Total_Melt'], '--', 
             label=f'{station} Melt', alpha=0.7)

plt.title('Normalized Flow Volume and Snow Melt by Year')
plt.xlabel('Year')
plt.ylabel('Normalized Value (z-score)')
plt.legend()

plt.tight_layout()
plt.savefig('flow_volume_analysis.png')

# Print analysis results
print("\nCorrelation Analysis (Total Melt vs Flow Volume):")
for station, (corr, p_value) in sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True):
    print(f"\n{station}:")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    # Calculate R-squared
    x = pd.to_numeric(melt_analysis[f'{station}_Total_Melt'], errors='coerce')
    y = pd.to_numeric(melt_analysis['Flow_Volume'], errors='coerce')
    mask = ~(np.isnan(x) | np.isnan(y))
    slope, intercept, r_value, _, _ = stats.linregress(x[mask], y[mask])
    r_squared = r_value ** 2
    print(f"R-squared: {r_squared:.3f}")

print("\nYearly Volume Analysis:")
print("\nYear    Flow_Volume  ", end="")
for station in stations:
    print(f"{station:>10}  ", end="")
print()

for year in year_labels:
    flow_vol = pd.to_numeric(melt_analysis['Flow_Volume'][year], errors='coerce')
    print(f"{year:4d}  {flow_vol:11.0f}  ", end="")
    
    for station in stations:
        melt_vol = pd.to_numeric(melt_analysis[f'{station}_Total_Melt'][year], errors='coerce')
        print(f"{melt_vol:10.1f}  ", end="")
    print()
