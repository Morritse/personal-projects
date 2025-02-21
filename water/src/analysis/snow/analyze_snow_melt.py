import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 15]
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# Load flow data
fnf_raw = pd.read_csv('fnf2010_2020.csv', header=None)
year_labels = list(range(2010, 2021))
fnf_raw.columns = year_labels

# Create continuous flow series (skip 2010 due to data quality)
flow_series = []
for year in year_labels[1:]:
    year_data = fnf_raw[year].dropna()
    dates = pd.date_range(start=f'{year-1}-10-01', periods=len(year_data))
    flow_series.append(pd.Series(year_data.values, index=dates))

flow_df = pd.concat(flow_series)

# Load and process snow data
stations = ['GRM', 'HNT', 'KSP', 'VLC']
snow_data = {}

for station in stations:
    # Read data
    df = pd.read_csv(f'{station}.csv')
    snow_df = pd.DataFrame({
        'Date': pd.to_datetime(df['DATE TIME'].str[:8], format='%Y%m%d'),
        'Water_Level': pd.to_numeric(df['VALUE'], errors='coerce')
    }).set_index('Date')
    
    # Filter out 2010 and calculate daily metrics
    snow_df = snow_df[snow_df.index.year >= 2011]
    
    # Calculate melt rate (negative change in water level)
    snow_df['Melt_Rate'] = -snow_df['Water_Level'].diff()
    
    # Calculate 7-day moving averages
    snow_df['MA7_Level'] = snow_df['Water_Level'].rolling(7).mean()
    snow_df['MA7_Melt'] = snow_df['Melt_Rate'].rolling(7).mean()
    
    snow_data[station] = snow_df

# Calculate flow volumes following significant melt events
melt_flow_stats = []
window_days = 90  # Days to look ahead for flow response

for station in stations:
    snow_df = snow_data[station]
    
    # Find significant melt events (top 10% of melt rates by year)
    for year in range(2011, 2021):
        year_data = snow_df[snow_df.index.year == year]
        if len(year_data) > 0:
            threshold = year_data['Melt_Rate'].quantile(0.9)
            melt_events = year_data[year_data['Melt_Rate'] > threshold]
            
            for date in melt_events.index:
                # Calculate subsequent flow volume
                flow_window = flow_df[date:date + pd.Timedelta(days=window_days)]
                if len(flow_window) > 0:
                    total_flow = flow_window.sum()
                    peak_flow = flow_window.max()
                    days_to_peak = (flow_window.idxmax() - date).days
                    
                    melt_flow_stats.append({
                        'Station': station,
                        'Date': date,
                        'Year': year,
                        'Month': date.month,
                        'Melt_Rate': melt_events.loc[date, 'Melt_Rate'],
                        'Snow_Level': melt_events.loc[date, 'Water_Level'],
                        'Total_Flow': total_flow,
                        'Peak_Flow': peak_flow,
                        'Days_to_Peak': days_to_peak
                    })

stats_df = pd.DataFrame(melt_flow_stats)

# Create figure
plt.figure(figsize=(15, 15))

# Plot 1: Snow levels and flow for all years
plt.subplot(3, 1, 1)

# Plot flow (30-day moving average)
flow_ma = flow_df.rolling(30).mean()
plt.plot(flow_df.index, flow_ma/1000, 'k-', label='Flow (30-day avg, k cfs)', alpha=0.7)

# Plot snow levels
for station in stations:
    plt.plot(snow_data[station].index, snow_data[station]['MA7_Level'], 
             '--', label=f'{station} Snow', alpha=0.5)

plt.title('Flow and Snow Water Content Over Time')
plt.xlabel('Date')
plt.ylabel('Flow (k cfs) / Snow Water Content (inches)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Response time analysis
plt.subplot(3, 1, 2)

# Calculate response time statistics by elevation
station_elevations = {
    'VLC': 'Low',
    'GRM': 'Mid',
    'HNT': 'High',
    'KSP': 'High'
}

response_stats = []
for station in stations:
    station_data = stats_df[stats_df['Station'] == station].copy()
    station_data['Season'] = pd.cut(station_data['Month'],
                                  bins=[0, 3, 6, 9, 12],
                                  labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = station_data[station_data['Season'] == season]
        if len(season_data) > 0:
            response_stats.append({
                'Station': station,
                'Elevation': station_elevations[station],
                'Season': season,
                'Avg_Response': season_data['Days_to_Peak'].mean(),
                'Event_Count': len(season_data),
                'Avg_Flow': season_data['Total_Flow'].mean()/1000
            })

response_df = pd.DataFrame(response_stats)

# Plot response times by elevation and season
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
elevations = ['Low', 'Mid', 'High']
colors = ['blue', 'green', 'red', 'orange']

for i, season in enumerate(seasons):
    season_data = response_df[response_df['Season'] == season]
    plt.scatter(season_data['Elevation'], season_data['Avg_Response'],
               s=season_data['Event_Count']*5, color=colors[i],
               alpha=0.6, label=season)

plt.title('Flow Response Time by Elevation and Season')
plt.xlabel('Station Elevation')
plt.ylabel('Average Days to Peak Flow')
plt.legend(title='Season')
plt.grid(True)

# Add size legend
sizes = [10, 50, 100]
legend_elements = [plt.scatter([], [], s=s*5, c='gray', alpha=0.3,
                             label=f'{s} events') for s in sizes]
plt.legend(handles=legend_elements, title='Event Count',
          loc='center left', bbox_to_anchor=(1, 0.5))

# Plot 3: Snow level vs flow volume by season
plt.subplot(3, 1, 3)

for i, season in enumerate(seasons):
    for station in stations:
        station_data = stats_df[stats_df['Station'] == station].copy()
        station_data['Season'] = pd.cut(station_data['Month'],
                                      bins=[0, 3, 6, 9, 12],
                                      labels=seasons)
        season_data = station_data[station_data['Season'] == season]
        
        if len(season_data) > 0:
            plt.scatter(season_data['Snow_Level'], 
                       season_data['Total_Flow']/1000,
                       color=colors[i], alpha=0.5,
                       label=f'{season} {station}' if station == stations[0] else None)

plt.title('Snow Water Content vs Flow Volume by Season')
plt.xlabel('Snow Water Content (inches)')
plt.ylabel('90-day Flow Volume (k cfs-days)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.savefig('snow_melt_analysis.png')

# Print analysis results
print("\nSnow Melt Analysis by Station:")
for station in stations:
    station_data = stats_df[stats_df['Station'] == station]
    print(f"\n{station}:")
    print(f"Number of significant melt events: {len(station_data)}")
    
    # Overall correlation
    corr, p_value = stats.pearsonr(station_data['Snow_Level'], 
                                 station_data['Total_Flow'])
    print(f"Snow level vs flow volume correlation: {corr:.3f} (p={p_value:.3f})")
    
    # Timing statistics
    print(f"Average days to peak flow: {station_data['Days_to_Peak'].mean():.1f}")
    print(f"Median days to peak flow: {station_data['Days_to_Peak'].median():.1f}")
    
    # Create copy for seasonal analysis
    seasonal_data = station_data.copy()
    seasonal_data.loc[:, 'Season'] = pd.cut(seasonal_data['Month'],
                                          bins=[0, 3, 6, 9, 12],
                                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    print("\nSeasonal Analysis:")
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = seasonal_data[seasonal_data['Season'] == season]
        if len(season_data) > 0:
            # Calculate transition metrics
            next_season = {'Winter': 'Spring', 'Spring': 'Summer',
                         'Summer': 'Fall', 'Fall': 'Winter'}[season]
            next_season_data = seasonal_data[seasonal_data['Season'] == next_season]
            
            print(f"\n  {season}:")
            print(f"    Events: {len(season_data)}")
            print(f"    Average melt rate: {season_data['Melt_Rate'].mean():.2f} in/day")
            print(f"    Average flow volume: {season_data['Total_Flow'].mean()/1000:.1f}k cfs-days")
            print(f"    Average days to peak: {season_data['Days_to_Peak'].mean():.1f}")
            
            if len(next_season_data) > 0:
                vol_change = ((next_season_data['Total_Flow'].mean() - 
                             season_data['Total_Flow'].mean()) / 
                            season_data['Total_Flow'].mean() * 100)
                print(f"    Volume change to {next_season}: {vol_change:+.1f}%")
