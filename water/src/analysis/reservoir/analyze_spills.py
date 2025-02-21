import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the data
df = pd.read_csv('water_years_base.csv')

# Find spill events (where reservoir level exceeds 522,500)
spills = df[df['Reservoir_Level'] > 522500].copy()

# Get all unique water years and their types
year_types = df.groupby('Water_Year')['Water_Year_Type'].first()
all_years = df['Water_Year'].unique()
spills_per_year = pd.Series(0, index=all_years)

# Update counts for years with spills
spill_counts = spills.groupby('Water_Year')['Reservoir_Level'].count()
spills_per_year.update(spill_counts)

# Sort by year
spills_per_year = spills_per_year.sort_index()

# Create the plot
plt.figure(figsize=(15, 8))
sns.barplot(x=range(len(spills_per_year)), y=spills_per_year.values)
plt.xticks(range(len(spills_per_year)), spills_per_year.index, rotation=45)
plt.title('Number of Days with Reservoir Level > 522,500 by Water Year', pad=20)
plt.ylabel('Number of Days')

# Add annotations for bars with spills
for i, (year, value) in enumerate(spills_per_year.items()):
    if value > 0:
        # Add spill days and water year type on top of bar
        plt.text(i, value, f"{int(value)}\nType {year_types[year]}", 
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('spill_analysis.png')

# Print summary
print("\nSpill Analysis Summary:")
print(f"Total number of spill days: {len(spills)}")
print("\nSpills by water year:")
for year, count in spills_per_year.items():
    if count > 0:
        year_spills = spills[spills['Water_Year'] == year]
        max_level = year_spills['Reservoir_Level'].max()
        max_date = year_spills.loc[year_spills['Reservoir_Level'].idxmax(), 'Date']
        first_date = year_spills['Date'].iloc[0]
        print(f"{year}: {int(count)} days")
        print(f"  First spill: {first_date}")
        print(f"  Max level: {int(max_level):,} on {max_date}")
