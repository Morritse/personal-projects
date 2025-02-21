import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Process FNF data into continuous time series
fnf_raw = pd.read_csv('fnf2010_2020.csv', header=None)
year_labels = list(range(2010, 2021))
fnf_raw.columns = year_labels

# Create continuous flow series
flow_series = []
for year in year_labels[1:]:  # Skip 2010
    year_data = fnf_raw[year].dropna()
    dates = pd.date_range(start=f'{year-1}-10-01', periods=len(year_data))
    flow_series.append(pd.Series(year_data.values, index=dates))

flow_df = pd.concat(flow_series)

# Process snow data
stations = ['GRM', 'HNT', 'KSP', 'VLC']
snow_series = {}

for station in stations:
    # Read data
    df = pd.read_csv(f'{station}.csv')
    snow_df = pd.DataFrame({
        'Date': pd.to_datetime(df['DATE TIME'].str[:8], format='%Y%m%d'),
        'Water_Level': pd.to_numeric(df['VALUE'], errors='coerce')
    })
    
    # Filter out 2010 data
    snow_df = snow_df[snow_df['Date'].dt.year >= 2011]
    
    # Create continuous series
    snow_series[station] = pd.Series(snow_df['Water_Level'].values, index=snow_df['Date'])

# Create figure
plt.figure(figsize=(15, 15))

# Plot 1: Cumulative volumes for example years
plt.subplot(3, 1, 1)

# Select years with contrasting melt patterns
example_years = [2017, 2015]  # high snow, low snow years
colors = ['blue', 'red']

for year, color in zip(example_years, colors):
    year_start = pd.Timestamp(f'{year-1}-10-01')
    year_end = pd.Timestamp(f'{year}-09-30')
    
    # Plot flow
    year_mask = (flow_df.index >= year_start) & (flow_df.index <= year_end)
    if year_mask.any():
        year_flow = flow_df[year_mask]
        # Calculate 30-day moving average
        flow_ma = year_flow.rolling(30).mean()
        plt.plot(year_flow.index, flow_ma, 
                color=color, linestyle='-', linewidth=2,
                label=f'{year} Flow (30-day avg)', alpha=0.7)
    
    # Plot snow melt rate for each station
    for station in stations:
        station_mask = (snow_series[station].index >= year_start) & \
                      (snow_series[station].index <= year_end)
        if station_mask.any():
            station_data = snow_series[station][station_mask]
            # Calculate melt rate (negative change in snow level)
            melt_rate = -station_data.diff().rolling(7).mean()  # 7-day smoothing
            plt.plot(station_data.index, melt_rate, 
                    color=color, linestyle='--', alpha=0.3,
                    label=f'{year} {station} Melt Rate' if station == stations[0] else None)

plt.title('Flow and Snow Melt Rate: High Snow (2017) vs Low Snow (2015)')
plt.xlabel('Date')
plt.ylabel('Flow (cfs) / Melt Rate (inches/day)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate peak analysis first
peak_analysis = []
for station in stations:
    snow = snow_series[station]
    
# Find periods of rapid melt
snow_series = pd.Series(snow)
melt_rate = -snow_series.diff().rolling(7).mean()  # 7-day smoothing
melt_peaks = melt_rate.rolling(window=30, center=True).max()
peak_melt_dates = melt_peaks[melt_peaks == melt_rate].index
    
    for peak_date in peak_dates:
        # Get subsequent 90-day flow volume
        flow_start = peak_date
        flow_end = flow_start + pd.Timedelta(days=90)
        if flow_end <= flow_df.index[-1]:
            # Calculate total flow volume and peak flow
            flow_window = flow_df[flow_start:flow_end]
            total_flow = flow_window.sum()
            peak_flow = flow_window.max()
            peak_flow_date = flow_window.idxmax()
            
            peak_analysis.append({
                'Station': station,
                'Melt_Date': peak_date,
                'Melt_Rate': melt_rate[peak_date],
                'Total_Flow': total_flow,
                'Peak_Flow': peak_flow,
                'Days_to_Peak': (peak_flow_date - peak_date).days
            })

peak_df = pd.DataFrame(peak_analysis)

# Plot 2: Seasonal correlation analysis
plt.subplot(3, 1, 2)

# Set up seasonal analysis
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
season_months = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5],
                'Summer': [6, 7, 8], 'Fall': [9, 10, 11]}

# Calculate seasonal correlations and volumes
seasonal_stats = []
for station in stations:
    station_data = peak_df[peak_df['Station'] == station].copy()
    station_data['Month'] = station_data['Peak_Date'].dt.month
    
    for season in seasons:
        season_data = station_data[station_data['Month'].isin(season_months[season])]
        if len(season_data) > 0:
            # Calculate correlation
            corr, p_value = stats.pearsonr(season_data['Peak_Snow'],
                                         season_data['Subsequent_Flow'])
            
            # Calculate average flow volume
            flow_vols = []
            for _, peak in season_data.iterrows():
                flow_start = peak['Peak_Date']
                flow_end = flow_start + pd.Timedelta(days=90)
                if flow_end <= flow_df.index[-1]:
                    flow_vol = flow_df[flow_start:flow_end].sum()
                    flow_vols.append(flow_vol)
            
            seasonal_stats.append({
                'Station': station,
                'Season': season,
                'Correlation': corr,
                'P_Value': p_value,
                'Avg_Flow_Volume': np.mean(flow_vols) if flow_vols else np.nan,
                'Peak_Count': len(season_data)
            })

# Create seasonal correlation plot
seasonal_df = pd.DataFrame(seasonal_stats)
for station in stations:
    station_data = seasonal_df[seasonal_df['Station'] == station]
    plt.plot(station_data['Season'], station_data['Correlation'], 
             'o-', label=station)

plt.title('Snow-Flow Correlation by Season')
plt.xlabel('Season')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 3: Peak snow to subsequent flow analysis
plt.subplot(3, 1, 3)

# Plot peak snow vs subsequent flow
for station in stations:
    station_data = peak_df[peak_df['Station'] == station]
    plt.scatter(station_data['Peak_Snow'], station_data['Subsequent_Flow'], 
               label=station, alpha=0.7)
    
    # Add trend line
    if len(station_data) > 1:
        z = np.polyfit(station_data['Peak_Snow'], station_data['Subsequent_Flow'], 1)
        p = np.poly1d(z)
        x = np.array([station_data['Peak_Snow'].min(), station_data['Peak_Snow'].max()])
        plt.plot(x, p(x), '--', alpha=0.5)

plt.title('Peak Snow Level vs Average Flow in Subsequent 90 Days')
plt.xlabel('Peak Snow Water Level (inches)')
plt.ylabel('Average Flow (cfs)')
plt.legend()

plt.tight_layout()
plt.savefig('snow_flow_sequence.png')

# Print analysis results
print("\nSeasonal Analysis Summary:")
for station in stations:
    station_data = peak_df[peak_df['Station'] == station]
    if len(station_data) > 1:
        corr, p_value = stats.pearsonr(station_data['Peak_Snow'], 
                                     station_data['Subsequent_Flow'])
        print(f"\n{station}:")
        print(f"Number of peaks analyzed: {len(station_data)}")
        print(f"Correlation: {corr:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        # Calculate average lag to max flow
        lags = []
        for _, peak in station_data.iterrows():
            peak_date = peak['Peak_Date']
            flow_start = peak_date
            flow_end = flow_start + pd.Timedelta(days=90)
            if flow_end <= flow_df.index[-1]:
                flow_window = flow_df[flow_start:flow_end]
                max_flow_date = flow_window.idxmax()
                lag = (max_flow_date - peak_date).days
                lags.append(lag)
        
        # Calculate seasonal statistics
        station_data['Season'] = pd.cut(station_data['Peak_Date'].dt.month,
                                      bins=[0, 3, 6, 9, 12],
                                      labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
# Print seasonal summary
for season in seasons:
    print(f"\n{season}:")
    season_data = seasonal_df[seasonal_df['Season'] == season]
    for station in stations:
        station_season = season_data[season_data['Station'] == station]
        if not station_season.empty:
            print(f"\n{station}:")
            print(f"  Correlation: {station_season['Correlation'].iloc[0]:.3f} " +
                  f"(p={station_season['P_Value'].iloc[0]:.3f})")
            print(f"  Average 90-day flow volume: {station_season['Avg_Flow_Volume'].iloc[0]/1000:.1f}k cfs-days")
            print(f"  Number of peaks: {station_season['Peak_Count'].iloc[0]}")
