import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta

def get_kondolf_flows():
    """Read Kondolf flows from CSV"""
    kondolf_df = pd.read_csv('kondolf.csv', header=None)
    # Each column corresponds to a water year type (1-6)
    return {i+1: kondolf_df[i].tolist() for i in range(6)}

def get_base_flows():
    """Return list of base flows for a standard year"""
    return [
        516, 417, 417, 417, 417, 417, 415, 415, 415, 413, 413, 409, 409, 409, 
        381, 353, 353, 353, 351, 351, 351, 351, 351, 351, 349, 349, 349, 349, 
        345, 345, 345, 345, 345, 345, 345, 345, 343, 343, 343, 343, 343, 343, 
        343, 343, 343, 343, 343, 343, 343, 343, 341, 341, 341, 341, 341, 341, 
        314, 298, 298, 298, 298, 298, 298, 298, 298, 298, 298, 270, 256, 256, 
        256, 256, 258, 258, 258, 258, 258, 258, 258, 258, 256, 222, 198, 198, 
        200, 200, 200, 200, 202, 202, 202, 202, 202, 202, 202, 202, 202, 206, 
        206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 208, 208, 
        208, 208, 206, 208, 208, 208, 208, 208, 208, 210, 210, 210, 210, 210, 
        210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 208, 208, 210, 210, 
        210, 210, 212, 212, 212, 212, 212, 214, 214, 214, 214, 210, 208, 208, 
        208, 208, 208, 208, 208, 208, 208, 208, 210, 210, 210, 210, 210, 210, 
        210, 290, 369, 480, 568, 568, 568, 568, 568, 568, 568, 568, 568, 568, 
        568, 568, 568, 568, 568, 568, 568, 568, 568, 568, 568, 566, 564, 564, 
        566, 570, 562, 548, 548, 546, 546, 544, 544, 544, 544, 544, 544, 544, 
        544, 540, 540, 540, 544, 544, 544, 546, 566, 597, 599, 601, 603, 605, 
        599, 599, 601, 603, 599, 599, 603, 593, 591, 597, 597, 601, 601, 601, 
        601, 601, 603, 603, 605, 607, 607, 607, 607, 605, 605, 605, 607, 609, 
        611, 605, 601, 601, 601, 599, 599, 599, 597, 599, 609, 611, 609, 609, 
        607, 603, 601, 605, 605, 605, 599, 595, 591, 591, 595, 597, 595, 593, 
        591, 591, 589, 591, 597, 601, 597, 597, 595, 597, 599, 522, 595, 593, 
        591, 589, 593, 587, 589, 593, 593, 645, 697, 693, 657, 599, 599, 599, 
        597, 597, 597, 595, 595, 595, 595, 595, 597, 498, 306, 490, 490, 494, 
        476, 447, 447, 447, 443, 443, 443, 443, 445, 437, 429, 429, 429, 429, 
        431, 429, 423, 417, 415, 415, 415, 415, 415, 415, 413, 415, 413, 413, 
        413, 413, 413, 409, 409, 409, 409, 409, 395, 385, 389, 389, 389, 389, 
        373, 367
    ]

def get_water_year_types():
    """Return list of water year types"""
    return [
        2, 1, 3, 3, 3, 1, 3, 1, 3, 4, 5, 2, 2, 3, 2, 3, 1, 3, 1, 2, 3, 3, 2, 
        2, 2, 4, 6, 1, 2, 1, 3, 1, 1, 2, 3, 1, 4, 4, 4, 4, 4, 4, 2, 3, 1, 2, 
        1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 4, 3, 3
    ]

def get_start_levels():
    """Return list of starting reservoir levels for each water year"""
    return [
        76184, 146076, 188214, 157069, 143151, 138302, 191888, 158371, 155992, 
        166656, 182785, 158419, 147041, 203511, 171050, 165161, 160139, 234907, 
        166731, 268201, 169046, 152358, 145731, 142247, 137540, 158395, 225906, 
        196688, 375548, 162313, 284434, 166058, 367709, 366500, 160844, 168945, 
        157406, 166606, 146191, 140377, 183024, 173989, 162683, 179073, 184378, 
        317334, 235652, 220769, 437551, 232129, 211003, 192654, 221186, 214591, 
        181490, 234504, 238330, 199171, 197634
    ]

def read_yearly_data(df, filename, scale_factor=None):
    """Read and reshape yearly data from CSV to match DataFrame structure"""
    # Read data (columns are water years)
    data_df = pd.read_csv(filename, header=None)
    
    # Name columns as water years starting from 1951
    data_df.columns = range(1951, 1951 + len(data_df.columns))
    
    # Initialize values list
    values = []
    
    # For each row in our main DataFrame
    for _, row in df.iterrows():
        year = row['WaterYear']
        day = row['DayOfYear']
        
        # If it's a leap year and after Feb 28, adjust day index
        if day > 59 and calendar.isleap(year):  # Feb 28 is day 59
            day_idx = day - 2  # Adjust for 0-based index and leap day
        else:
            day_idx = day - 1  # Adjust for 0-based index
            
        # Get value if year exists and day index is valid
        if year in data_df.columns and day_idx < len(data_df):
            value = data_df[year].iloc[day_idx]
            if scale_factor is not None:
                value = round(value * scale_factor)
            values.append(value)
        else:
            values.append(None)
    
    return values

def read_inflow_data(df):
    """Read and reshape inflow data from CSV to match DataFrame structure"""
    return read_yearly_data(df, 'inflow.csv', scale_factor=1.98347)

def read_diversion_data(df):
    """Read and reshape diversion data from CSV to match DataFrame structure"""
    return read_yearly_data(df, 'diversion.csv')

def create_water_year_index(start_year, num_years):
    """Create index for multiple water years handling leap years"""
    
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    dates = []
    water_year_days = []
    water_years = []
    
    for year in range(start_year, start_year + num_years):
        # Water year starts October 1 of previous calendar year
        start_date = datetime(year - 1, 10, 1)
        
        # Check if we'll encounter Feb 29 in this water year
        has_leap_day = is_leap_year(year)
        num_days = 366 if has_leap_day else 365
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            dates.append(current_date)
            water_year_days.append(day + 1)  # 1-based day of water year
            water_years.append(year)
    
    # Create DataFrame with water year information
    df = pd.DataFrame({
        'Date': dates,
        'WaterYear': water_years,
        'DayOfYear': water_year_days
    })
    
    # Add month and day columns for reference
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Add water year type (repeats for each day in the year)
    year_types = get_water_year_types()
    df['WaterYearType'] = df['WaterYear'].apply(lambda x: year_types[x - start_year])
    
    # Add inflow and diversion data
    print(f"\nDataFrame length: {len(df)}")
    inflows = read_inflow_data(df)
    print(f"Inflow data length: {len(inflows)}")
    df['Inflow'] = inflows
    
    diversions = read_diversion_data(df)
    print(f"Diversion data length: {len(diversions)}")
    df['Diversion'] = diversions
    
    # Calculate running total for each water year
    start_levels = get_start_levels()
    df['TotalIn'] = None
    
    # Initialize first day of each water year with start level
    df.loc[(df['Month'] == 10) & (df['Day'] == 1), 'TotalIn'] = \
        df.loc[(df['Month'] == 10) & (df['Day'] == 1), 'WaterYear'].apply(lambda x: start_levels[x - start_year])
    
    # Calculate running total within each water year
    for year in df['WaterYear'].unique():
        year_mask = df['WaterYear'] == year
        year_data = df[year_mask].copy()
        
        # Start with initial value
        current_total = year_data.loc[year_data.index[0], 'TotalIn']
        
        # Calculate running total
        for idx in year_data.index[1:]:  # Skip first day since it's already set
            current_total += year_data.loc[idx, 'Inflow']
            df.loc[idx, 'TotalIn'] = current_total
    
    # Add Kondolf flows based on water year type
    kondolf_flows = get_kondolf_flows()
    df['KondolfFlow'] = df.apply(
        lambda row: kondolf_flows[row['WaterYearType']][row['DayOfYear'] - 1] if row['DayOfYear'] <= 365 
        else kondolf_flows[row['WaterYearType']][364],  # Use last day's flow for Feb 29
        axis=1
    )
    
    # Add base flows last, handling leap years
    base_flows = get_base_flows()
    df['BaseFlow'] = df.apply(
        lambda row: base_flows[row['DayOfYear'] - 1] if row['DayOfYear'] <= 365 
        else base_flows[364],  # Use last day's flow for Feb 29
        axis=1
    )
    
    return df

# Example usage:
if __name__ == "__main__":
    # Create index for water years 1951-2009
    df = create_water_year_index(1951, 59)  # 59 years from 1951 to 2009 inclusive
    
    # Show water year transition example
    print("\nExample: Water Year Transition (1951-1952)")
    transitions = df[
        ((df['Month'] == 9) & (df['Day'].isin([29, 30]))) |  # End of water year
        ((df['Month'] == 10) & (df['Day'].isin([1, 2])))     # Start of water year
    ].sort_values('Date').head(8)  # Show first two transitions
    print(transitions[['Date', 'WaterYear', 'WaterYearType', 'TotalIn', 'Inflow', 'BaseFlow', 'KondolfFlow', 'Diversion']])
    
    # Show leap year handling
    print("\nExample: Leap Year 1952 (February)")
    feb_leap = df[df['WaterYear'] == 1952].query('Month == 2')
    print(feb_leap[['Date', 'WaterYear', 'WaterYearType', 'TotalIn', 'Inflow', 'BaseFlow', 'KondolfFlow', 'Diversion']])
    
    print("\nExample: Non-Leap Year 1953 (February)")
    feb_normal = df[df['WaterYear'] == 1953].query('Month == 2')
    print(feb_normal[['Date', 'WaterYear', 'WaterYearType', 'TotalIn', 'Inflow', 'BaseFlow', 'KondolfFlow', 'Diversion']])
    
    # Reorder columns
    column_order = ['Date', 'WaterYear', 'DayOfYear', 'Month', 'Day', 'WaterYearType', 
                    'TotalIn', 'Inflow', 'BaseFlow', 'KondolfFlow', 'Diversion']
    df = df[column_order]
    
    # Save template
    df.to_csv('historical_data_template.csv', index=False)
    print("\nTemplate saved to historical_data_template.csv")
    print(f"Total rows: {len(df)}")
    
    # Display structure
    print("\nDataFrame Structure:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nTemplate saved to historical_data_template.csv")
