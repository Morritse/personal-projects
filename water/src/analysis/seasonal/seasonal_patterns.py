import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# Read data
df = pd.read_csv('water_years_base.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Create monthly averages by year type
monthly_type_avg = df.groupby(['Water_Year_Type', 'Month']).agg({
    'Reservoir_Level': 'mean',
    'Inflow': 'mean'
}).reset_index()

# Plot average reservoir levels by month for each year type
plt.subplot(2, 1, 1)
sns.lineplot(data=monthly_type_avg, x='Month', y='Reservoir_Level', 
             hue='Water_Year_Type', marker='o')
plt.axhline(y=522500, color='red', linestyle='--', label='Spill Level')
plt.title('Average Reservoir Level by Month and Year Type')
plt.xlabel('Month')
plt.ylabel('Reservoir Level')

# Plot average inflow by month for each year type
plt.subplot(2, 1, 2)
sns.lineplot(data=monthly_type_avg, x='Month', y='Inflow', 
             hue='Water_Year_Type', marker='o')
plt.title('Average Inflow by Month and Year Type')
plt.xlabel('Month')
plt.ylabel('Inflow')

plt.tight_layout()
plt.savefig('seasonal_patterns.png')

# Print seasonal statistics
print("\nSeasonal Statistics:")
print("\n1. Peak Reservoir Months by Year Type:")
for year_type in sorted(df['Water_Year_Type'].unique()):
    type_data = monthly_type_avg[monthly_type_avg['Water_Year_Type'] == year_type]
    peak_month = type_data.loc[type_data['Reservoir_Level'].idxmax()]
    print(f"\nType {year_type}:")
    print(f"Peak Month: {int(peak_month['Month'])}")
    print(f"Average Level: {peak_month['Reservoir_Level']:.0f}")
    print(f"Average Inflow: {peak_month['Inflow']:.0f}")
