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

# Get yearly peak flows
yearly_peaks = {}
for year in year_labels:
    yearly_peaks[year] = fnf_raw[year].max()

# Process snow data
stations = ['GRM', 'HNT', 'KSP', 'VLC']
snow_volumes = pd.DataFrame(index=year_labels)

for station in stations:
    # Read data
    df = pd.read_csv(f'{station}.csv')
    snow_df = pd.DataFrame({
        'Date': pd.to_datetime(df['DATE TIME'].str[:8], format='%Y%m%d'),
        'Water_Level': pd.to_numeric(df['VALUE'], errors='coerce')
    })
    
    # Get yearly peak snow water levels
    snow_df['Year'] = snow_df['Date'].dt.year
    yearly_max = snow_df.groupby('Year')['Water_Level'].max()
    snow_volumes[f'{station}_Volume'] = yearly_max

# Add peak flows to the DataFrame
snow_volumes['Peak_Flow'] = pd.Series(yearly_peaks)

# Create figure
plt.figure(figsize=(15, 10))

# Plot 1: Scatter plots of peak flow vs snow volume for each station
plt.subplot(2, 1, 1)
correlations = {}

for station in stations:
    x = snow_volumes[f'{station}_Volume']
    y = snow_volumes['Peak_Flow']
    
    # Convert to numeric and handle NaN
    x_clean = pd.to_numeric(x.dropna(), errors='coerce')
    y_clean = pd.to_numeric(y.dropna(), errors='coerce')
    
    # Remove any remaining NaN values
    mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
    x_clean = x_clean[mask]
    y_clean = y_clean[mask]
    
    # Calculate correlation and p-value
    corr, p_value = stats.pearsonr(x_clean, y_clean)
    correlations[station] = (corr, p_value)
    
    plt.scatter(x_clean, y_clean, alpha=0.7, label=f'{station} (r={corr:.3f})')
    
    # Add trend line using clean data
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    plt.plot(x_clean, p(x_clean), '--', alpha=0.5)

plt.title('Peak Flow vs Peak Snow Water Volume by Station')
plt.xlabel('Peak Snow Water Volume (inches)')
plt.ylabel('Peak Flow (cfs)')
plt.legend()

# Plot 2: Year-by-year comparison
plt.subplot(2, 1, 2)

# Normalize values for comparison
normalized_data = pd.DataFrame(index=year_labels)

# Convert and normalize peak flow
normalized_data['Peak_Flow'] = pd.to_numeric(snow_volumes['Peak_Flow'], errors='coerce')
normalized_data['Peak_Flow'] = (normalized_data['Peak_Flow'] - normalized_data['Peak_Flow'].mean()) / normalized_data['Peak_Flow'].std()

# Convert and normalize station volumes
for station in stations:
    col = f'{station}_Volume'
    normalized_data[col] = pd.to_numeric(snow_volumes[col], errors='coerce')
    normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()

# Plot normalized values
years = normalized_data.index
plt.plot(years, normalized_data['Peak_Flow'], 'k-', label='Peak Flow', linewidth=2)
for station in stations:
    plt.plot(years, normalized_data[f'{station}_Volume'], '--', 
             label=f'{station} Volume', alpha=0.7)

plt.title('Normalized Peak Flow and Snow Water Volume by Year')
plt.xlabel('Year')
plt.ylabel('Normalized Value (z-score)')
plt.legend()

plt.tight_layout()
plt.savefig('snow_volume_analysis.png')

# Print correlation analysis
print("\nCorrelation Analysis (Peak Snow Volume vs Peak Flow):")
for station, (corr, p_value) in correlations.items():
    print(f"\n{station}:")
    print(f"Correlation: {corr:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    # Calculate R-squared using clean data
    x_clean = pd.to_numeric(snow_volumes[f'{station}_Volume'].dropna(), errors='coerce')
    y_clean = pd.to_numeric(snow_volumes['Peak_Flow'].dropna(), errors='coerce')
    mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
    slope, intercept, r_value, _, _ = stats.linregress(x_clean[mask], y_clean[mask])
    r_squared = r_value ** 2
    print(f"R-squared: {r_squared:.3f}")

# Print yearly comparison with proper numeric handling
print("\nYearly Peak Values:")
print("\nYear    Peak_Flow    ", end="")
for station in stations:
    print(f"{station:>10}  ", end="")
print()

for year in year_labels:
    peak_flow = pd.to_numeric(snow_volumes['Peak_Flow'][year], errors='coerce')
    print(f"{year:4d}  {peak_flow:10.0f}    ", end="")
    
    for station in stations:
        snow_val = pd.to_numeric(snow_volumes[f'{station}_Volume'][year], errors='coerce')
        print(f"{snow_val:10.1f}  ", end="")
    print()

# Print additional analysis
print("\nKey Findings:")
print("1. Correlation Strength by Station (strongest to weakest):")
sorted_stations = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)
for station, (corr, p_value) in sorted_stations:
    print(f"   {station}: r={corr:.3f} (p={p_value:.3f})")

print("\n2. Years with Highest Snow Volume vs Peak Flow:")
for station in stations:
    col = f'{station}_Volume'
    max_snow_year = snow_volumes[col].astype(float).idxmax()
    max_snow = snow_volumes[col].astype(float).max()
    flow_that_year = pd.to_numeric(snow_volumes['Peak_Flow'][max_snow_year])
    print(f"\n{station}:")
    print(f"   Max snow year: {max_snow_year} ({max_snow:.1f} inches)")
    print(f"   Peak flow that year: {flow_that_year:.0f} cfs")
