import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def analyze_water_year(year):
    """Analyze reservoir operations for a specific water year"""
    # Read the historical data
    df = pd.read_csv('historical_data_template.csv')
    
    # Filter for specific water year and reset index
    df = df[df['WaterYear'] == year].copy()
    df = df.reset_index(drop=True)  # Reset index to 0-based continuous numbers
    
    if len(df) == 0:
        print(f"Error: No data found for water year {year}")
        return
    
    print(f"Days in water year: {len(df)}")  # Debug output
    if len(df) != 365:  # Check for full year
        print(f"Warning: Expected 365 days, got {len(df)} days")

    # Constants
    SPILL_LEVEL = 525000  # Reservoir spill level
    
    # Initialize columns
    df['Level'] = 0.0
    df['Spill'] = 0.0
    
    # Set first day's level to initial TotalIn value
    df.iloc[0, df.columns.get_loc('Level')] = df.iloc[0]['TotalIn']
    
    # Calculate level for each subsequent day
    for i in range(1, len(df)):
        prev_level = df.iloc[i-1]['Level']  # Start with actual level (≤ 525k)
        inflow = df.iloc[i]['Inflow']
        baseflow = df.iloc[i]['BaseFlow']
        kondolf = df.iloc[i]['KondolfFlow']
        diversion = df.iloc[i]['Diversion']
        
        # Calculate available water
        available = prev_level + inflow  # Start with what we have
        
        # If we're above spill level, spill the excess
        if available > SPILL_LEVEL:
            spill = available - SPILL_LEVEL
            available = SPILL_LEVEL
        else:
            spill = 0
        
        # Then subtract outflows
        new_level = available - (baseflow + kondolf + diversion)
        new_level = min(new_level, SPILL_LEVEL)  # Cap at spill level
        
        # Debug output for first few days and spill days
        if i < 5 or spill > 0:
            total_outflow = baseflow + kondolf + diversion
            print(f"Day {i+1}:")
            print(f"  Previous level: {prev_level:,.0f}")
            print(f"  + Inflow: {inflow:,.0f}")
            print(f"  = Available: {available:,.0f}")
            if spill > 0:
                print(f"  - Spill: {spill:,.0f}")
                print(f"  = After spill: {SPILL_LEVEL:,.0f}")
            print(f"  - Outflow: {total_outflow:,.0f}")
            print(f"  = Final level: {new_level:,.0f}")
            print()
        
        # Store values
        df.iloc[i, df.columns.get_loc('Level')] = new_level
        df.iloc[i, df.columns.get_loc('Spill')] = spill
    
    # Analysis with detailed debug
    print("\nDetailed Analysis:")
    print("-" * 50)
    
    # Starting point
    start_level = df.iloc[0]['TotalIn']
    print(f"Starting level: {start_level:,.0f}")
    
    # Total flows
    total_inflow = df['Inflow'].sum()
    total_base_flow = df['BaseFlow'].sum()
    total_kondolf = df['KondolfFlow'].sum()
    total_diversion = df['Diversion'].sum()
    total_outflow = total_base_flow + total_kondolf + total_diversion
    
    print(f"\nFlow Totals:")
    print(f"  Total inflow: {total_inflow:,.0f}")
    print(f"  Total outflow: {total_outflow:,.0f}")
    print(f"    Base flow: {total_base_flow:,.0f}")
    print(f"    Kondolf: {total_kondolf:,.0f}")
    print(f"    Diversion: {total_diversion:,.0f}")
    
    # Daily spill analysis
    total_spill_daily = df['Spill'].sum()
    spill_days = (df['Spill'] > 0).sum()
    print(f"\nDaily Spill Analysis:")
    print(f"  Spill days: {spill_days}")
    print(f"  Total daily spill: {total_spill_daily:,.0f}")
    
    # Show first few spills with day numbers
    spill_mask = df['Spill'] > 0
    spill_days_list = df[spill_mask].index.tolist()[:5]  # Get first 5 spill days
    if spill_days_list:
        print("\nFirst few spill events:")
        for day_idx in spill_days_list:
            row = df.iloc[day_idx]
            day_num = day_idx + 1  # Convert to 1-based day number
            print(f"  Day {day_num}:")
            print(f"    Inflow: {row['Inflow']:,.0f}")
            print(f"    Outflow: {row['BaseFlow'] + row['KondolfFlow'] + row['Diversion']:,.0f}")
            print(f"    Spill: {row['Spill']:,.0f}")
            print(f"    Level: {row['Level']:,.0f}")
    
    # Get final level and water year type
    final_level = df.iloc[-1]['Level']
    water_year_type = df['WaterYearType'].iloc[0]
    
    print(f"Reservoir Analysis - Water Year {year} (Type {water_year_type})")
    print("-" * 50)
    print(f"Starting level: {start_level:,.0f} AF")
    print(f"Final level: {final_level:,.0f} AF")
    print(f"Total inflow: {total_inflow:,.0f} AF")
    print(f"Total outflow: {total_outflow:,.0f} AF")
    print(f"Total annual spill: {total_spill_daily:,.0f} AF")
    print("-" * 50)
    
    print(f"Detailed flows:")
    print(f"  Base flow: {total_base_flow:,.0f} AF")
    print(f"  Kondolf flow: {total_kondolf:,.0f} AF")
    print(f"  Diversion: {total_diversion:,.0f} AF")
    # Calculate maximum daily net inflow (inflow - outflows)
    df['NetInflow'] = df['Inflow'] - (df['BaseFlow'] + df['KondolfFlow'] + df['Diversion'])
    max_daily_net = df['NetInflow'].max()
    
    print(f"\nSpill days: {spill_days}")
    print(f"Maximum daily net inflow: {max_daily_net:,.0f} AF")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Plot reservoir level
    plt.plot(range(len(df)), df['Level'], label='Reservoir Level', color='blue')
    
    # Plot spill level line
    plt.axhline(y=SPILL_LEVEL, color='r', linestyle='--', label='Spill Level')
    
    # Plot daily spill amounts as bars
    spill_mask = df['Spill'] > 0
    if spill_mask.any():
        plt.bar(range(len(df)), df['Spill'].where(spill_mask, 0),
               bottom=SPILL_LEVEL, color='red', alpha=0.5,
               label='Daily Spill')
    
    # Add labels and title
    plt.xlabel('Day of Water Year')
    plt.ylabel('Acre-Feet')
    plt.title(f'Water Year {year} Reservoir Operation (Type {water_year_type})')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f'historical_analysis_{year}.png')
    print(f"\nPlot saved to historical_analysis_{year}.png")
    
    # Save detailed results
    df.to_csv(f'historical_analysis_{year}.csv', index=False)
    print(f"Results saved to historical_analysis_{year}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze reservoir operations for a specific water year')
    parser.add_argument('year', type=int, help='Water year to analyze (1951-2009)')
    args = parser.parse_args()
    
    analyze_water_year(args.year)
