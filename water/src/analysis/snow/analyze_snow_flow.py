import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Process FNF data
fnf_raw = pd.read_csv('fnf2010_2020.csv', header=None)
year_labels = list(range(2010, 2021))  # 2010-2020
fnf_raw.columns = year_labels

# Convert FNF data to long format with dates
fnf_data = []
for year in year_labels:
    year_data = fnf_raw[year].dropna()
    dates = [(pd.Timestamp(year-1, 10, 1) + pd.Timedelta(days=i)).strftime('%Y%m%d') 
            for i in range(len(year_data))]
    fnf_data.append(pd.DataFrame({
        'Date': dates,
        'Full_Flow': pd.to_numeric(year_data.values, errors='coerce')
    }))
fnf_df = pd.concat(fnf_data, ignore_index=True)
fnf_df['Date'] = pd.to_datetime(fnf_df['Date'], format='%Y%m%d')

# Read and process snow water level data for each station
stations = ['GRM', 'HNT', 'KSP', 'VLC']
snow_dfs = {}
for station in stations:
    # Read only needed columns
    df = pd.read_csv(f'{station}.csv')
    snow_dfs[station] = pd.DataFrame({
        'Date': pd.to_datetime(df['DATE TIME'].str[:8], format='%Y%m%d'),
        'Water_Level': pd.to_numeric(df['VALUE'], errors='coerce')
    })

# Create figure
plt.figure(figsize=(15, 10))

# Plot 1: Time series of all data
plt.subplot(2, 1, 1)
plt.plot(fnf_df['Date'], fnf_df['Full_Flow'], label='Full Flow', alpha=0.7)
for station in stations:
    plt.plot(snow_dfs[station]['Date'], snow_dfs[station]['Water_Level'], 
             label=f'{station} Snow Water', alpha=0.7)
plt.title('Full Flow and Snow Water Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

# Calculate correlations
correlations = {}
for station in stations:
    # Merge snow data with full flow data on date
    merged = pd.merge(fnf_df, snow_dfs[station], on='Date', how='inner')
    corr = merged['Full_Flow'].corr(merged['Water_Level'])
    correlations[station] = corr

# Plot 2: Correlation comparison
plt.subplot(2, 1, 2)
stations_sorted = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)
corr_values = [correlations[station] for station in stations_sorted]
plt.bar(stations_sorted, corr_values)
plt.title('Correlation between Snow Water Level and Full Flow by Station')
plt.xlabel('Station')
plt.ylabel('Correlation Coefficient')

# Add correlation values on top of bars
for i, v in enumerate(corr_values):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('snow_flow_analysis.png')

# Print detailed statistics
print("\nCorrelations with Full Flow:")
for station in stations_sorted:
    print(f"{station}: {correlations[station]:.3f}")

# Calculate monthly correlations
print("\nMonthly correlations:")
for station in stations:
    merged = pd.merge(fnf_df, snow_dfs[station], on='Date', how='inner')
    merged['Month'] = merged['Date'].dt.month
    monthly_corr = merged.groupby('Month').apply(
        lambda x: x['Full_Flow'].corr(x['Water_Level'])
    )
    print(f"\n{station}:")
    for month in range(1, 13):
        if month in monthly_corr.index:
            print(f"Month {month}: {monthly_corr[month]:.3f}")
