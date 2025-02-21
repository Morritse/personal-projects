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

# Plot 1: Example years with melt rates
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
        plt.plot(year_flow.index, flow_ma/1000, 
                color=color, linestyle='-', linewidth=2,
                label=f'{year} Flow (30-day avg, k cfs)', alpha=0.7)
    
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
                    label=f'{year} {station} Melt Rate (in/day)' if station == stations[0] else None)

plt.title('Flow and Snow Melt Rate: High Snow (2017) vs Low Snow (2015)')
plt.xlabel('Date')
plt.ylabel('Flow (k cfs) / Melt Rate (inches/day)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Calculate melt analysis
peak_analysis = []
for station in stations:
    # Calculate melt rate series
    snow = snow_series[station]
    melt_rate = -snow.diff().rolling(7).mean()  # 7-day smoothing
    
    # Find significant melt events (top 10% of melt rates in each year)
    for year in range(2011, 2021):
        year_start = pd.Timestamp(f'{year-1}-10-01')
        year_end = pd.Timestamp(f'{year}-09-30')
        year_mask = (melt_rate.index >= year_start) & (melt_rate.index <= year_end)
        
        if year_mask.any():
            year_melt = melt_rate[year_mask]
            threshold = year_melt.quantile(0.9)  # Top 10% of melt rates
            melt_events = year_melt[year_melt > threshold]
            
            for event_date in melt_events.index:
                # Calculate subsequent flow metrics
                flow_start = event_date
                flow_end = flow_start + pd.Timedelta(days=90)
                
                if flow_end <= flow_df.index[-1]:
                    flow_window = flow_df[flow_start:flow_end]
                    total_flow = flow_window.sum()
                    peak_flow = flow_window.max()
                    peak_flow_date = flow_window.idxmax()
                    
                    peak_analysis.append({
                        'Station': station,
                        'Year': year,
                        'Melt_Date': event_date,
                        'Melt_Rate': melt_rate[event_date],
                        'Total_Flow': total_flow,
                        'Peak_Flow': peak_flow,
                        'Days_to_Peak': (peak_flow_date - event_date).days
                    })

peak_df = pd.DataFrame(peak_analysis)

# Plot 2: Melt rate vs subsequent flow volume
plt.subplot(3, 1, 2)

for station in stations:
    station_data = peak_df[peak_df['Station'] == station]
    plt.scatter(station_data['Melt_Rate'], station_data['Total_Flow']/1000, 
               label=station, alpha=0.7)
    
    # Add trend line
    if len(station_data) > 1:
        z = np.polyfit(station_data['Melt_Rate'], station_data['Total_Flow']/1000, 1)
        p = np.poly1d(z)
        x = np.array([station_data['Melt_Rate'].min(), station_data['Melt_Rate'].max()])
        plt.plot(x, p(x), '--', alpha=0.5)

plt.title('Melt Rate vs Subsequent 90-day Flow Volume')
plt.xlabel('Melt Rate (inches/day)')
plt.ylabel('90-day Flow Volume (k cfs-days)')
plt.legend()
plt.grid(True)

# Plot 3: Seasonal timing patterns
plt.subplot(3, 1, 3)

# Add month column for all data
peak_df['Month'] = peak_df['Melt_Date'].dt.month

# Calculate monthly statistics
monthly_stats = []
for station in stations:
    station_data = peak_df[peak_df['Station'] == station]
    for month in range(1, 13):
        month_data = station_data[station_data['Month'] == month]
        if len(month_data) > 0:
            monthly_stats.append({
                'Station': station,
                'Month': month,
                'Avg_Days': month_data['Days_to_Peak'].mean(),
                'Event_Count': len(month_data)
            })

monthly_df = pd.DataFrame(monthly_stats)

# Plot average days to peak by month
for station in stations:
    station_data = monthly_df[monthly_df['Station'] == station]
    plt.plot(station_data['Month'], station_data['Avg_Days'], 
             'o-', label=f'{station}', alpha=0.7)

plt.title('Average Days to Peak Flow by Month')
plt.xlabel('Month')
plt.ylabel('Average Days to Peak Flow')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)

# Add size-coded points for event counts
for station in stations:
    station_data = monthly_df[monthly_df['Station'] == station]
    sizes = station_data['Event_Count'] * 20  # Scale point sizes
    plt.scatter(station_data['Month'], station_data['Avg_Days'],
               s=sizes, alpha=0.3)

plt.tight_layout()
plt.savefig('snow_melt_analysis.png')

# Print analysis results
print("\nMelt Rate Analysis by Station:")
for station in stations:
    station_data = peak_df[peak_df['Station'] == station]
    print(f"\n{station}:")
    print(f"Number of significant melt events: {len(station_data)}")
    
    # Correlation between melt rate and total flow
    corr, p_value = stats.pearsonr(station_data['Melt_Rate'], 
                                 station_data['Total_Flow'])
    print(f"Correlation (melt rate vs flow volume): {corr:.3f} (p={p_value:.3f})")
    
    # Timing statistics
    print(f"Average days to peak flow: {station_data['Days_to_Peak'].mean():.1f}")
    print(f"Median days to peak flow: {station_data['Days_to_Peak'].median():.1f}")
    
    # Flow volume statistics
    print(f"Average 90-day flow volume: {station_data['Total_Flow'].mean()/1000:.1f}k cfs-days")
    
    # Seasonal distribution
    station_data['Month'] = station_data['Melt_Date'].dt.month
    station_data['Season'] = pd.cut(station_data['Month'],
                                  bins=[0, 3, 6, 9, 12],
                                  labels=['Winter', 'Spring', 'Summer', 'Fall'])
    season_counts = station_data['Season'].value_counts()
    print("\nSeasonal distribution of melt events:")
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        if season in season_counts:
            print(f"  {season}: {season_counts[season]} events")
