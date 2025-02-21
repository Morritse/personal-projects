import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file, skipping empty columns
df = pd.read_csv('water_model.csv', thousands=',', usecols=[0,1,2,3,4])
df.columns = ['Level', 'Inflow', 'Fixed1', 'Fixed2', 'Variable']

# Clean up any whitespace in column names
df.columns = df.columns.str.strip()

# Constants
MAX_LEVEL = 525000  # Maximum reservoir level
MIN_FLOW = 500  # Minimum environmental flow
MAX_WITHDRAWAL =13000  # Maximum daily withdrawal capacity
MAX_DAILY_CHANGE = 0.02  # Maximum 2% change per day
SMOOTHING_WINDOW = 14    # 14-day moving average window
SMOOTHING_FACTOR = 0.9   # Exponential smoothing factor (higher = smoother)

def calculate_inflow_patterns(inflows, window_short=60, window_long=120):
    """Analyze inflow patterns over different time periods"""
    if len(inflows) < window_long:
        return {
            'recent_avg': np.mean(inflows),
            'is_wet_season': False,
            'max_recent': 0,
            'pre_snowmelt': False,
            'day_of_year': 0
        }
    
    # Calculate averages over different periods
    recent_avg = np.mean(inflows[-window_short:])
    long_term_avg = np.mean(inflows[-window_long:])
    
    # Look at maximum recent inflow to detect potential spikes
    max_recent = max(inflows[-window_short:])
    
    # Determine if we're in wet season by comparing recent to long-term
    is_wet_season = recent_avg > (long_term_avg * 1.05)
    
    # Calculate day of year (assuming inflows list starts at day 0)
    day_of_year = len(inflows) % 365
    
    # Detect pre-snowmelt period (days 100-150, typically late winter)
    pre_snowmelt = 100 <= day_of_year <= 150
    
    return {
        'recent_avg': recent_avg,
        'is_wet_season': is_wet_season,
        'max_recent': max_recent,
        'pre_snowmelt': pre_snowmelt,
        'day_of_year': day_of_year
    }

def get_target_withdrawal(current_level, historical_variable, inflow_history):
    """Calculate target withdrawal based on conditions"""
    
    # Get inflow patterns using longer windows
    patterns = calculate_inflow_patterns(inflow_history)
    recent_avg = patterns['recent_avg']
    is_wet_season = patterns['is_wet_season']
    max_recent = patterns['max_recent']
    
    # Calculate how full the reservoir is (as a percentage)
    fullness = (current_level - 300000) / (MAX_LEVEL - 300000)
    fullness = max(0, min(1, fullness))  # Clamp between 0 and 1
    
    # Start with historical withdrawal
    target = historical_variable
    
    # Calculate base increase based on conditions
    if fullness > 0.85:
        # High level - maximum withdrawal
        target = MAX_WITHDRAWAL
    elif fullness > 0.65:
        # Moderately high - sustained high withdrawals
        if is_wet_season or patterns['pre_snowmelt']:
            # In wet season or pre-snowmelt, be very aggressive
            target = historical_variable + (max_recent * 0.95)
            
            # During pre-snowmelt, maintain even higher withdrawals
            if patterns['pre_snowmelt']:
                target *= 1.1  # 10% higher during pre-snowmelt
        else:
            # Not in wet season, but still high withdrawals
            target = historical_variable + (recent_avg * 0.9)
            
        # Additional increase near top
        if fullness > 0.75:
            # Scale up to maximum over 75-85% range
            level_factor = (fullness - 0.75) / 0.1
            target = target + (MAX_WITHDRAWAL - target) * level_factor
            
        # During pre-snowmelt, be more aggressive at high levels
        if patterns['pre_snowmelt'] and fullness > 0.8:
            target = max(target, MAX_WITHDRAWAL * 0.95)  # At least 95% of max
    elif fullness > 0.45:
        # Moderate level - proactive response
        if is_wet_season or patterns['pre_snowmelt']:
            # In wet season or pre-snowmelt, maintain elevated withdrawals
            base_withdrawal = historical_variable + (recent_avg * 0.85)  # More aggressive
            if patterns['pre_snowmelt']:
                base_withdrawal *= 1.1  # Extra 10% during pre-snowmelt
            
            # Scale smoothly from 45-65% full
            level_factor = (fullness - 0.45) / 0.2
            target = historical_variable + (base_withdrawal - historical_variable) * level_factor
        else:
            # Not in wet season but still increase
            base_withdrawal = historical_variable + (recent_avg * 0.7)  # More aggressive baseline
            level_factor = (fullness - 0.45) / 0.2
            target = historical_variable + (base_withdrawal - historical_variable) * level_factor
    elif is_wet_season:
        # Lower level but wet season - gradual increase
        base_withdrawal = historical_variable + (recent_avg * 0.5)
        target = historical_variable + (base_withdrawal - historical_variable) * fullness
    
    # Ensure within limits
    return min(MAX_WITHDRAWAL, target)

def calculate_smoothed_withdrawal(target, prev_targets, prev_withdrawal, historical, current_level):
    """Apply multiple layers of smoothing to the target withdrawal"""
    smoothed = target
    
    # Calculate reservoir fullness
    fullness = (current_level - 300000) / (MAX_LEVEL - 300000)
    fullness = max(0, min(1, fullness))
    
    # Reduce smoothing at high water levels
    if fullness > 0.8:
        # Use shorter window and less smoothing when high
        window = max(3, int(SMOOTHING_WINDOW * (1 - fullness)))
        smoothing_factor = max(0.5, SMOOTHING_FACTOR * (1 - fullness))
        max_daily_pct = min(0.05, MAX_DAILY_CHANGE * (2 - fullness))
    else:
        # Normal smoothing during regular operations
        window = SMOOTHING_WINDOW
        smoothing_factor = SMOOTHING_FACTOR
        max_daily_pct = MAX_DAILY_CHANGE
    
    # Apply moving average if we have enough history
    if len(prev_targets) >= window:
        recent_targets = prev_targets[-window:]
        moving_avg = np.mean(recent_targets)
        # Blend with more weight to moving average during normal operations
        weight = 0.75 if fullness <= 0.8 else 0.25
        smoothed = (target * (1 - weight) + moving_avg * weight)
    
    # Apply exponential smoothing
    if prev_withdrawal is not None:
        smoothed = smoothing_factor * prev_withdrawal + (1 - smoothing_factor) * smoothed
    
    # Apply gradual rate limiting
    max_change = prev_withdrawal * max_daily_pct
    if smoothed > prev_withdrawal:
        smoothed = min(smoothed, prev_withdrawal + max_change)
    else:
        smoothed = max(smoothed, prev_withdrawal - max_change)
    
    # Never go below historical
    return max(historical, min(MAX_WITHDRAWAL, smoothed))

def simulate_reservoir(df):
    """Simulate reservoir behavior with recommended withdrawals"""
    results = []
    current_level = df.iloc[0]['Level']
    inflow_history = []
    prev_withdrawal = df.iloc[0]['Variable']
    prev_targets = []
    
    for index, row in df.iterrows():
        # Update inflow history
        inflow_history.append(row['Inflow'])
        
        # Calculate target withdrawal
        target = get_target_withdrawal(
            current_level,
            row['Variable'],
            inflow_history
        )
        prev_targets.append(target)
        
        # Apply smoothing with current level context
        recommended = calculate_smoothed_withdrawal(
            target,
            prev_targets,
            prev_withdrawal,
            row['Variable'],
            current_level
        )
        
        # Ensure within absolute limits
        recommended = max(row['Variable'], min(MAX_WITHDRAWAL, recommended))
        
        # Calculate next day's level and spill
        uncapped_level = current_level + row['Inflow'] - row['Fixed1'] - row['Fixed2'] - recommended
        spill = max(0, uncapped_level - MAX_LEVEL)
        next_level = min(uncapped_level, MAX_LEVEL)
        
        # Calculate net inflow (excluding variable withdrawals)
        net_inflow = row['Inflow'] - row['Fixed1'] - row['Fixed2']
        
        results.append({
            'Day': index,
            'Level': current_level,
            'Net_Inflow': net_inflow,
            'Inflow': row['Inflow'],
            'Fixed1': row['Fixed1'],
            'Fixed2': row['Fixed2'],
            'Recommended_Withdrawal': recommended,
            'Next_Level': next_level,
            'Spill': spill,
            'Total_Level': uncapped_level  # For visualization
        })
        
        # Update for next iteration
        current_level = next_level
        prev_withdrawal = recommended
    
    return pd.DataFrame(results)

def plot_results(results_df):
    """Plot reservoir levels and withdrawals over time"""
    plt.figure(figsize=(15, 10))
    
    # Create subplot for reservoir level
    plt.subplot(2, 1, 1)
    # Plot actual reservoir level
    plt.plot(results_df['Day'], results_df['Level'], label='Actual Level', color='blue')
    
    # Plot spill level only where it occurs
    spill_mask = results_df['Spill'] > 0
    if spill_mask.any():
        plt.plot(results_df.loc[spill_mask, 'Day'], 
                results_df.loc[spill_mask, 'Total_Level'],
                color='red', alpha=0.5, label='Spill Level')
        
    plt.axhline(y=MAX_LEVEL, color='r', linestyle='--', label='Maximum Level')
    
    # Fill spill area
    plt.fill_between(results_df['Day'], 
                     results_df['Level'],
                     results_df['Total_Level'],
                     where=(results_df['Spill'] > 0),
                     color='red', alpha=0.3, label='Spill Volume')
    plt.ylabel('Reservoir Level (AF)')
    plt.title('Reservoir Level Over Time')
    plt.legend()
    plt.grid(True)
    
    # Create subplot for withdrawals
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Day'], results_df['Recommended_Withdrawal'], 
             label='Recommended Withdrawal', color='green')
    plt.axhline(y=MAX_WITHDRAWAL, color='r', linestyle='--', 
                label=f'Max Withdrawal ({MAX_WITHDRAWAL:,} AF/day)')
    plt.ylabel('Withdrawal (AF/day)')
    plt.xlabel('Day')
    plt.title('Variable Withdrawals Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reservoir_analysis.png')
    plt.close()

# Run simulation and analysis
results_df = simulate_reservoir(df)

# Calculate total yearly volumes
total_inflow = df['Inflow'].sum()  # Use original dataframe for totals
total_fixed_withdrawals = df['Fixed1'].sum() + df['Fixed2'].sum()
total_variable_withdrawals = results_df['Recommended_Withdrawal'].sum()

# Calculate total spill volume and days
total_spill_volume = results_df['Spill'].sum()
spill_days = len(results_df[results_df['Spill'] > 0])

print("\nReservoir Analysis")
print("-" * 50)
print(f"Total yearly inflow: {total_inflow:,.0f} AF")
print(f"Total fixed withdrawals: {total_fixed_withdrawals:,.0f} AF")
print(f"Total variable withdrawals: {total_variable_withdrawals:,.0f} AF")
print(f"Total spill volume: {total_spill_volume:,.0f} AF")
print(f"\nNumber of spill days: {spill_days}")

# Calculate withdrawal statistics
avg_withdrawal = results_df['Recommended_Withdrawal'].mean()
max_withdrawal = results_df['Recommended_Withdrawal'].max()
print(f"\nAverage recommended withdrawal: {avg_withdrawal:,.0f} AF/day")
print(f"Maximum recommended withdrawal: {max_withdrawal:,.0f} AF/day")

# Find maximum net inflow
max_rise = results_df['Net_Inflow'].max()
print(f"Maximum daily net inflow: {max_rise:,.0f} AF")

# Save results
results_df.to_csv('reservoir_analysis.csv', index=False)
plot_results(results_df)
print("\nResults saved to reservoir_analysis.csv")
print("Plots saved to reservoir_analysis.png")
