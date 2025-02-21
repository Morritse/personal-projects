import pandas as pd
from datetime import datetime, timedelta

# Create dates for all years
dates = []
water_years = []
start_date = datetime(1950, 10, 1)
end_date = datetime(2020, 9, 30)

current_date = start_date
while current_date <= end_date:
    dates.append(current_date.strftime('%m/%d/%Y'))
    water_year = current_date.year + 1 if current_date.month >= 10 else current_date.year
    water_years.append(f"WY {water_year}")
    current_date += timedelta(days=1)

# Create water year type mapping (1951-2022)
water_year_types = {
    1951: 2, 1952: 1, 1953: 3, 1954: 3, 1955: 3, 1956: 1, 1957: 3, 1958: 1, 1959: 3, 1960: 4,
    1961: 5, 1962: 2, 1963: 2, 1964: 3, 1965: 2, 1966: 3, 1967: 1, 1968: 3, 1969: 1, 1970: 2,
    1971: 3, 1972: 3, 1973: 2, 1974: 2, 1975: 2, 1976: 4, 1977: 6, 1978: 1, 1979: 2, 1980: 1,
    1981: 3, 1982: 1, 1983: 1, 1984: 2, 1985: 3, 1986: 1, 1987: 4, 1988: 4, 1989: 4, 1990: 4,
    1991: 4, 1992: 4, 1993: 2, 1994: 3, 1995: 1, 1996: 2, 1997: 1, 1998: 1, 1999: 2, 2000: 2,
    2001: 3, 2002: 3, 2003: 3, 2004: 3, 2005: 1, 2006: 1, 2007: 4, 2008: 3, 2009: 3, 2010: 2,
    2011: 1, 2012: 3, 2013: 4, 2014: 5, 2015: 6, 2016: 3, 2017: 1, 2018: 3, 2019: 1, 2020: 4
}

# Create base DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Water_Year': water_years
})

# Add water year type (only for years we have types for)
df['Water_Year_Type'] = df['Water_Year'].apply(
    lambda x: water_year_types.get(int(x.replace('WY ', '')), None)
)

# Reorder columns to put Water_Year_Type after Water_Year
df = df[['Date', 'Water_Year', 'Water_Year_Type']]

def process_data(filename, value_column, scale_factor=None):
    # Read data
    data_df = pd.read_csv(filename, header=None)
    year_labels = list(range(1951, 1951 + len(data_df.columns)))
    data_df.columns = year_labels
    
    # Process each year
    all_data = []
    for year in range(1951, 1951 + len(data_df.columns)):
        year_data = data_df[year].dropna()  # Drop empty values
        
        # Convert to numeric, replacing any non-numeric values with NaN
        year_data = pd.to_numeric(year_data, errors='coerce')
        
        # Apply scaling if provided and round inflow values
        if scale_factor is not None:
            year_data = (year_data * scale_factor).round(0)
            
        dates = [(datetime(year-1, 10, 1) + timedelta(days=i)).strftime('%m/%d/%Y') 
                for i in range(len(year_data))]
        all_data.append(pd.DataFrame({
            'Date': dates,
            value_column: year_data.values
        }))
    
    return pd.concat(all_data, ignore_index=True)

# Process all data files
reservoir_data = process_data('historical_reservoir_level.csv', 'Reservoir_Level')
inflow_data = process_data('inflow.csv', 'Inflow', scale_factor=1.98347)
actual_sjr_data = process_data('actual_sjr_out.csv', 'Actual_SJR_Out', scale_factor=1.98347)
full_flow_data = process_data('fullflow.csv', 'Full_Flow', scale_factor=1.98347)
madera_data = process_data('madera_out.csv', 'Madera_Out', scale_factor=1.98347)
friant_data = process_data('friant_out.csv', 'Friant_Out', scale_factor=1.98347)

# Read required out data
required_out = pd.read_csv('required_out.csv', header=None)

# Merge all data with base structure
df = df.merge(reservoir_data, on='Date', how='left')
df = df.merge(inflow_data, on='Date', how='left')
df = df.merge(actual_sjr_data, on='Date', how='left')
df = df.merge(full_flow_data, on='Date', how='left')
df = df.merge(madera_data, on='Date', how='left')
df = df.merge(friant_data, on='Date', how='left')

# Print first few rows of required out data to verify structure
print("\nFirst few rows of required out data:")
print(required_out.head())

# Add required out column based on water year type
df['Required_Out'] = None
for year in df['Water_Year'].unique():
    year_mask = df['Water_Year'] == year
    year_data = df[year_mask].copy()
    if pd.notnull(year_data['Water_Year_Type'].iloc[0]):
        type_num = int(year_data['Water_Year_Type'].iloc[0])
        print(f"\nYear: {year}, Type: {type_num}")
        print(f"First value from column {type_num-1}: {required_out.iloc[0, type_num-1]}")
        for i, idx in enumerate(year_data.index):
            df.at[idx, 'Required_Out'] = required_out.iloc[i % len(required_out), type_num - 1]

# Reorder columns
df = df[['Date', 'Water_Year', 'Water_Year_Type', 'Reservoir_Level', 'Inflow', 'Required_Out', 'Actual_SJR_Out', 'Full_Flow', 'Madera_Out', 'Friant_Out']]

# Save result with proper CSV formatting
df.to_csv('water_years_base.csv', index=False, sep=',')

# Verification
print("\nVerification:")
print(f"Total rows in final dataset: {len(df)}")
print("\nFirst day (Oct 1) values for each water year:")
oct1_data = df[df['Date'].str.contains('10/01/')]
print(oct1_data.head(10))

print("\nNumber of days per water year:")
print(df.groupby('Water_Year').size().head())

# Check a few dates around March 1st for verification
print("\nChecking March 1st transitions:")
for year in [1952, 1953]:  # Check both leap and non-leap year
    dates = [f'02/28/{year}', f'02/29/{year}', f'03/01/{year}', f'03/02/{year}']
    print(f"\nYear {year}:")
    print(df[df['Date'].isin(dates)])

# Check a type 5 year (1961)
print("\nChecking a Type 5 year (1961):")
type5_data = df[df['Water_Year'] == 'WY 1961'].head()
print(type5_data)
